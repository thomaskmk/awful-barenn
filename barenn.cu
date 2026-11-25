// Versão CUDA do treino com mini-batch (issues #14, #17 e #18).
//
// O treino roda inteiro na GPU: os dados e os pesos iniciais vão para a GPU
// uma vez, antes do treino, e os pesos finais e as perdas voltam uma vez,
// depois. Um único kernel de um bloco treina todas as épocas, com os pesos na
// memória compartilhada do começo ao fim (#18). Leitura dos dados,
// inicialização dos pesos e avaliação no teste continuam na CPU, iguais às do
// barenn.cpp.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

// Interrompe com a mensagem da CUDA se uma chamada da API falhar.
#define CUDA_CHECK(chamada)                                                        \
    do {                                                                           \
        cudaError_t erro_cuda = (chamada);                                         \
        if (erro_cuda != cudaSuccess) {                                            \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro_cuda) + \
                                " em " #chamada);                                  \
        }                                                                          \
    } while (0)

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Larguras das camadas ocultas: a rede é 64 -> 32 -> 16 -> 10.
constexpr int oculta1 = 32;
constexpr int oculta2 = 16;

// Threads do bloco que treina a rede. 1.024 é o máximo por bloco; o tamanho
// usado vem da variável de ambiente BLOCO (padrão: threads_padrao).
constexpr int max_threads_por_bloco = 1024;
constexpr int threads_padrao = 1024;

// Floats na memória compartilhada por amostra do batch: a entrada, as
// ativações das três camadas e os deltas das três camadas.
constexpr int floats_por_amostra = entradas + 2 * (oculta1 + oculta2 + classes);

// Floats da cópia dos pesos na memória compartilhada. Cada linha ocupa
// in_features + 1 posições: no forward, as 32 threads de um warp leem a mesma
// coluna de 32 linhas, e com passo 64 todas cairiam no mesmo banco.
constexpr int floats_pesos =
    oculta1 * (entradas + 1) + oculta2 * (oculta1 + 1) + classes * (oculta2 + 1);

// Floats dos vieses das três camadas na memória compartilhada.
constexpr int floats_vieses = oculta1 + oculta2 + classes;

// ---------------------------------------------------------------------------
// CPU: dados, parâmetros iniciais e avaliação (os mesmos do barenn.cpp)
// ---------------------------------------------------------------------------

// Parâmetros de uma camada na CPU. A inicialização é a do barenn.cpp, então os
// pesos iniciais são idênticos. O forward serve só para a avaliação.
struct Camada {
    int in_features;
    int out_features;
    bool final;
    // Matriz em um único bloco, linha por linha: weights[i * in_features + j]
    // liga a entrada j ao neurônio de saída i.
    vector<float> weights;
    vector<float> bias;

    Camada(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size * input_size), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    // Forward de uma amostra: input tem in_features valores; out, out_features.
    void Forward(const float* input, float* out) const {
        for (int i = 0; i < out_features; ++i) {
            const float* w = weights.data() + i * in_features; // linha i
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * w[j];
            }
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : max(0.0f, sum);
        }
    }
};

// Conjunto de imagens em um único bloco de memória: a imagem s ocupa as
// posições [s * entradas, (s + 1) * entradas) de imagens.
struct Dados {
    vector<float> imagens;
    vector<int> rotulos;

    size_t tamanho() const { return rotulos.size(); }
    const float* imagem(size_t s) const { return imagens.data() + s * entradas; }
};

Dados ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    Dados dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        int lidos = 0;
        for (; lidos < entradas; ++lidos) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            dados.imagens.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        int rotulo;
        if (lidos != entradas ||
            !(campos >> rotulo) ||
            rotulo < 0 || rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.rotulos.push_back(rotulo);
    }
    if (arquivo.bad() || dados.tamanho() == 0) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais.
float perda(const float* escores, int rotulo) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
    }
    return total;
}

void avaliar(const Camada& l1, const Camada& l2, const Camada& l3, const Dados& teste) {
    // Só forward, com os pesos trazidos da GPU.
    vector<float> a1(l1.out_features), a2(l2.out_features), saida(l3.out_features);
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    for (size_t i = 0; i < teste.tamanho(); ++i) {
        const int rotulo = teste.rotulos[i];
        l1.Forward(teste.imagem(i), a1.data());
        l2.Forward(a1.data(), a2.data());
        l3.Forward(a2.data(), saida.data());
        const float* escores = saida.data();
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(max_element(escores, escores + classes) - escores);
        bool acertou = previsto == rotulo;
        acertos += acertou;
        ++confusoes[rotulo][previsto];
        perda_total += perda(escores, rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.tamanho()
         << "% | " << acertos << " acertos em " << teste.tamanho() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.tamanho() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const Confusao& a, const Confusao& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// GPU: treino em um único bloco
//
// As etapas que eram kernels separados na v3.2 viram funções do mesmo kernel,
// separadas por barreiras (__syncthreads). Cada etapa distribui seus elementos
// entre as threads do bloco, de blockDim.x em blockDim.x, então o tamanho do
// bloco é só um parâmetro. Os pesos e os vetores do batch ficam na memória
// compartilhada: o valor k da amostra s fica em [s * largura + k]. Cada soma é
// feita na mesma ordem do barenn.cpp; as diferenças de arredondamento vêm da
// contração em FMA feita pelo nvcc.
// ---------------------------------------------------------------------------

// Copia os pesos de uma camada da memória global para a compartilhada, com
// passo in_features + 1 entre as linhas. A leitura da global é coalescida.
__device__ void copiar_pesos(const float* W, float* w, int in_features, int out_features) {
    for (int k = threadIdx.x; k < out_features * in_features; k += blockDim.x) {
        w[(k / in_features) * (in_features + 1) + k % in_features] = W[k];
    }
}

// Grava os pesos de uma camada de volta na memória global, sem o passo extra.
__device__ void gravar_pesos(const float* w, float* W, int in_features, int out_features) {
    for (int k = threadIdx.x; k < out_features * in_features; k += blockDim.x) {
        W[k] = w[(k / in_features) * (in_features + 1) + k % in_features];
    }
}

// Copia n floats de um vetor para outro (vieses, entrada do batch).
__device__ void copiar(const float* origem, float* destino, int n) {
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        destino[k] = origem[k];
    }
}

// Forward de uma camada. Um elemento por neurônio de uma amostra:
// idx = s * out_features + i. A linha i dos pesos começa em W + i * passo.
__device__ void forward_camada(const float* entrada, const float* W, int passo, const float* b,
                               float* saida, int n, int in_features, int out_features,
                               bool final) {
    for (int idx = threadIdx.x; idx < n * out_features; idx += blockDim.x) {
        const int s = idx / out_features;
        const int i = idx % out_features;
        const float* x = entrada + s * in_features;
        const float* w = W + i * passo; // linha i
        float sum = b[i];
        for (int j = 0; j < in_features; ++j) {
            sum += x[j] * w[j];
        }
        // A mesma expressão de max(0.0f, sum) na CPU.
        saida[idx] = final ? sum : (0.0f < sum ? sum : 0.0f);
    }
}

// Perda e delta da camada de saída, uma amostra por vez: a perda da amostra
// soma as dez classes, como na CPU. O delta da saída é o próprio erro.
__device__ void perda_saida(const float* escores, const int* rotulos,
                            float* delta, float* perda_amostra, int n) {
    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        float total = 0;
        for (int i = 0; i < classes; ++i) {
            const float erro = escores[s * classes + i] - (i == rotulos[s] ? 1.0f : 0.0f);
            total += 0.5f * erro * erro;
            delta[s * classes + i] = erro;
        }
        perda_amostra[s] = total;
    }
}

// Delta da camada anterior. Um elemento por entrada de uma amostra:
// idx = s * in_features + j. O grad_in[j] soma delta[i] * W[i][j] na ordem de i,
// e a derivada da ReLU da camada anterior deixa passar só onde a ativação é
// positiva, o que equivale a z > 0. A linha i dos pesos começa em W + i * passo.
__device__ void propagar_delta(const float* delta, const float* W, int passo,
                               const float* ativacao_anterior, float* delta_anterior,
                               int n, int in_features, int out_features) {
    for (int idx = threadIdx.x; idx < n * in_features; idx += blockDim.x) {
        const int s = idx / in_features;
        const int j = idx % in_features;
        const float* d = delta + s * out_features;
        float grad = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            grad += d[i] * W[i * passo + j];
        }
        delta_anterior[idx] = grad * (ativacao_anterior[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// Atualização dos parâmetros. Um elemento por peso e um por viés: cada um soma
// as correções das n amostras do batch, na ordem das amostras, e aplica a
// média. Nenhuma outra thread escreve no mesmo parâmetro. A linha i dos pesos
// começa em W + i * passo.
__device__ void atualizar_camada(float* W, int passo, float* b, const float* delta,
                                 const float* entrada, int n, int in_features, int out_features,
                                 float taxa, float inv) {
    const int pesos = out_features * in_features;
    for (int idx = threadIdx.x; idx < pesos + out_features; idx += blockDim.x) {
        if (idx < pesos) {
            const int i = idx / in_features;
            const int j = idx % in_features;
            float soma = 0.0f;
            for (int s = 0; s < n; ++s) {
                soma -= taxa * (delta[s * out_features + i] * entrada[s * in_features + j]);
            }
            W[i * passo + j] += soma * inv;
        } else {
            const int i = idx - pesos;
            float soma = 0.0f;
            for (int s = 0; s < n; ++s) {
                soma -= taxa * delta[s * out_features + i];
            }
            b[i] += soma * inv;
        }
    }
}

// Treina todas as épocas em um único bloco. Os pesos e os vieses ficam na
// memória compartilhada do começo ao fim: são lidos da memória global uma vez,
// no início, e gravados de volta uma vez, no fim. Entre um batch e o seguinte,
// uma barreira substitui o lançamento de um novo kernel. A entrada, as
// ativações e os deltas do batch também ficam na compartilhada, com espaço
// para capacidade amostras.
__global__ void __launch_bounds__(max_threads_por_bloco)
treinar_kernel(const float* imagens, const int* rotulos,
               float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
               float* perda_amostra, int N, int batch_size, int capacidade, int epocas,
               float taxa) {
    extern __shared__ float compartilhada[];
    float* w1 = compartilhada;                    // oculta1 × (entradas + 1)
    float* w2 = w1 + oculta1 * (entradas + 1);    // oculta2 × (oculta1 + 1)
    float* w3 = w2 + oculta2 * (oculta1 + 1);     // classes × (oculta2 + 1)
    float* v1 = w3 + classes * (oculta2 + 1);     // oculta1 (vieses)
    float* v2 = v1 + oculta1;                     // oculta2
    float* v3 = v2 + oculta2;                     // classes
    float* x = v3 + classes;                      // capacidade × entradas
    float* a1 = x + capacidade * entradas;        // capacidade × oculta1
    float* a2 = a1 + capacidade * oculta1;        // capacidade × oculta2
    float* a3 = a2 + capacidade * oculta2;        // capacidade × classes (escores)
    float* d1 = a3 + capacidade * classes;        // capacidade × oculta1
    float* d2 = d1 + capacidade * oculta1;        // capacidade × oculta2
    float* d3 = d2 + capacidade * oculta2;        // capacidade × classes

    // Os pesos e os vieses são lidos da memória global uma vez, para o treino todo.
    copiar_pesos(W1, w1, entradas, oculta1);
    copiar_pesos(W2, w2, oculta1, oculta2);
    copiar_pesos(W3, w3, oculta2, classes);
    copiar(b1, v1, oculta1);
    copiar(b2, v2, oculta2);
    copiar(b3, v3, classes);

    for (int epoca = 0; epoca < epocas; ++epoca) {
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (int inicio = 0; inicio < N; inicio += batch_size) {
            // O último batch pode ser menor que batch_size.
            const int n = min(batch_size, N - inicio);
            const float inv = 1.0f / n;

            // A entrada do batch é lida da memória global uma vez e
            // reaproveitada no forward e na atualização da primeira camada.
            copiar(imagens + size_t(inicio) * entradas, x, n * entradas);
            __syncthreads();

            forward_camada(x, w1, entradas + 1, v1, a1, n, entradas, oculta1, false);
            __syncthreads();
            forward_camada(a1, w2, oculta1 + 1, v2, a2, n, oculta1, oculta2, false);
            __syncthreads();
            forward_camada(a2, w3, oculta2 + 1, v3, a3, n, oculta2, classes, true);
            __syncthreads();
            perda_saida(a3, rotulos + inicio, d3, perda_amostra + size_t(epoca) * N + inicio, n);
            __syncthreads();

            // Propaga o delta da última camada até a primeira, com os pesos
            // ainda sem a atualização deste batch.
            propagar_delta(d3, w3, oculta2 + 1, a2, d2, n, oculta2, classes);
            __syncthreads();
            propagar_delta(d2, w2, oculta1 + 1, a1, d1, n, oculta1, oculta2);
            __syncthreads();

            // Uma atualização por batch, com a média das correções das n
            // amostras. As três camadas não dependem umas das outras.
            atualizar_camada(w3, oculta2 + 1, v3, d3, a2, n, oculta2, classes, taxa, inv);
            atualizar_camada(w2, oculta1 + 1, v2, d2, a1, n, oculta1, oculta2, taxa, inv);
            atualizar_camada(w1, entradas + 1, v1, d1, x, n, entradas, oculta1, taxa, inv);
            // O próximo batch só começa com todos os pesos atualizados.
            __syncthreads();
        }
    }

    // Os pesos finais voltam para a memória global uma vez, no fim.
    gravar_pesos(w1, W1, entradas, oculta1);
    gravar_pesos(w2, W2, oculta1, oculta2);
    gravar_pesos(w3, W3, oculta2, classes);
    copiar(v1, b1, oculta1);
    copiar(v2, b2, oculta2);
    copiar(v3, b3, classes);
}

// ---------------------------------------------------------------------------
// GPU: memória
// ---------------------------------------------------------------------------

float* alocar_floats(size_t n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

// Parâmetros de uma camada na GPU.
struct CamadaGPU {
    float* W;
    float* b;

    explicit CamadaGPU(const Camada& c)
        : W(alocar_floats(c.weights.size())), b(alocar_floats(c.bias.size())) {}

    void liberar() {
        cudaFree(W);
        cudaFree(b);
    }

    void enviar(const Camada& c) {
        CUDA_CHECK(cudaMemcpy(W, c.weights.data(), c.weights.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, c.bias.data(), c.bias.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void trazer(Camada& c) const {
        CUDA_CHECK(cudaMemcpy(c.weights.data(), W, c.weights.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c.bias.data(), b, c.bias.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        // Tamanho do batch pela variável de ambiente BATCH_SIZE.
        // Sem a variável, B = 1 (SGD por amostra, igual ao código base).
        int batch_size = 1;
        if (const char* valor = getenv("BATCH_SIZE")) {
            size_t lidos = 0;
            try { batch_size = stoi(valor, &lidos); } catch (const logic_error&) { lidos = 0; }
            if (lidos == 0 || lidos != string(valor).size() || batch_size < 1) {
                throw runtime_error("BATCH_SIZE deve ser um inteiro >= 1");
            }
        }

        // Taxa de aprendizado pela variável de ambiente TAXA.
        // Sem a variável, 0,02 (a do código base). Com a média das correções
        // do batch, batches maiores precisam de taxas maiores.
        float taxa = 0.02f;
        if (const char* valor = getenv("TAXA")) {
            size_t lidos = 0;
            try { taxa = stof(valor, &lidos); } catch (const logic_error&) { lidos = 0; }
            if (lidos == 0 || lidos != string(valor).size() || !(taxa > 0.0f)) {
                throw runtime_error("TAXA deve ser um número > 0");
            }
        }

        // Threads do bloco pela variável de ambiente BLOCO: um múltiplo de 32
        // (um warp) de 32 a 1.024.
        int bloco = Barenn::threads_padrao;
        if (const char* valor = getenv("BLOCO")) {
            size_t lidos = 0;
            try { bloco = stoi(valor, &lidos); } catch (const logic_error&) { lidos = 0; }
            if (lidos == 0 || lidos != string(valor).size() || bloco < 32 ||
                bloco > Barenn::max_threads_por_bloco || bloco % 32 != 0) {
                throw runtime_error("BLOCO deve ser um múltiplo de 32 entre 32 e 1024");
            }
        }

        constexpr int epocas = 51;
        const int N = static_cast<int>(treino.tamanho());
        const int max_batch = min(batch_size, N);

        // Os pesos, os vieses e os vetores de um batch precisam caber na
        // memória compartilhada de um bloco.
        const size_t bytes_parametros = (Barenn::floats_pesos + Barenn::floats_vieses) * sizeof(float);
        const size_t bytes_por_amostra = Barenn::floats_por_amostra * sizeof(float);
        int compartilhada_por_bloco = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&compartilhada_por_bloco,
                                          cudaDevAttrMaxSharedMemoryPerBlock, 0));
        const int max_batch_gpu =
            static_cast<int>((compartilhada_por_bloco - bytes_parametros) / bytes_por_amostra);
        if (max_batch > max_batch_gpu) {
            throw runtime_error("BATCH_SIZE deve ser <= " + to_string(max_batch_gpu) +
                                " nesta GPU: o batch precisa caber na memória compartilhada de um bloco");
        }
        const size_t bytes_compartilhada = bytes_parametros + max_batch * bytes_por_amostra;

        // Rede densa com dimensões 64 -> 32 -> 16 -> 10, inicializada na CPU.
        Barenn::Camada l1(Barenn::entradas, Barenn::oculta1);
        Barenn::Camada l2(Barenn::oculta1, Barenn::oculta2);
        Barenn::Camada l3(Barenn::oculta2, Barenn::classes, /*final=*/true);

        // Memória da GPU, alocada uma vez: dados de treino, parâmetros e a
        // perda de cada amostra em cada época.
        Barenn::CamadaGPU g1(l1), g2(l2), g3(l3);
        float* imagens = Barenn::alocar_floats(treino.imagens.size());
        int* rotulos = nullptr;
        CUDA_CHECK(cudaMalloc(&rotulos, treino.rotulos.size() * sizeof(int)));
        float* perda_amostra = Barenn::alocar_floats(size_t(epocas) * N);

        // A CUDA carrega cada kernel no primeiro uso. Carrega antes do
        // cronômetro, para que o carregamento não entre na medição.
        cudaFuncAttributes atributos;
        CUDA_CHECK(cudaFuncGetAttributes(&atributos, Barenn::treinar_kernel));

        // Ida: dados e pesos iniciais, uma vez, fora do cronômetro do treino.
        const auto inicio_ida = chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(imagens, treino.imagens.data(), treino.imagens.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rotulos, treino.rotulos.data(), treino.rotulos.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        g1.enviar(l1);
        g2.enviar(l2);
        g3.enviar(l3);
        const chrono::duration<double> tempo_ida = chrono::steady_clock::now() - inicio_ida;

        // A medição inclui forward, perda, backward e atualização de todas as épocas.
        // Um único kernel de um bloco faz o treino inteiro.
        const auto inicio = chrono::steady_clock::now();
        Barenn::treinar_kernel<<<1, bloco, bytes_compartilhada>>>(
            imagens, rotulos, g1.W, g1.b, g2.W, g2.b, g3.W, g3.b,
            perda_amostra, N, batch_size, max_batch, epocas, taxa);
        // O lançamento é assíncrono: espera a GPU terminar antes de parar.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Volta: pesos finais e perdas, uma vez, fora do cronômetro do treino.
        vector<float> perdas_por_amostra(size_t(epocas) * N);
        const auto inicio_volta = chrono::steady_clock::now();
        g1.trazer(l1);
        g2.trazer(l2);
        g3.trazer(l3);
        CUDA_CHECK(cudaMemcpy(perdas_por_amostra.data(), perda_amostra,
                              perdas_por_amostra.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const chrono::duration<double> tempo_volta = chrono::steady_clock::now() - inicio_volta;

        g1.liberar();
        g2.liberar();
        g3.liberar();
        cudaFree(imagens);
        cudaFree(rotulos);
        cudaFree(perda_amostra);

        // Perda média de cada época, somando as amostras na ordem do arquivo,
        // como o perda_total do barenn.cpp.
        array<double, epocas> perdas{};
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (int s = 0; s < N; ++s) {
                perda_total += perdas_por_amostra[size_t(epoca) * N + s];
            }
            perdas[epoca] = perda_total / N;
        }

        // Imprime as métricas e roda a avaliação na CPU com os pesos trazidos.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.tamanho() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        cout << setprecision(3)
             << "Tempo de copias (fora do treino): ida " << tempo_ida.count() * 1e3
             << " ms, volta " << tempo_volta.count() * 1e3 << " ms\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
