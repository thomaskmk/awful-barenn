// Versão CUDA do treino com mini-batch (issues #14 e #17).
//
// O treino roda inteiro na GPU: os dados e os pesos iniciais vão para a GPU
// uma vez, antes do treino, e os pesos finais e as perdas voltam uma vez,
// depois. Cada batch é treinado por um único kernel de um bloco (#17).
// Leitura dos dados, inicialização dos pesos e avaliação no teste continuam
// na CPU, iguais às do barenn.cpp.
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

// Threads do bloco que treina um batch. 1.024 é o máximo por bloco, e a maior
// etapa (forward da primeira camada com B = 32) tem 32 × 32 = 1.024 saídas.
constexpr int threads_por_bloco = 1024;

// Floats na memória compartilhada por amostra do batch: a entrada, as
// ativações das três camadas e os deltas das três camadas.
constexpr int floats_por_amostra = entradas + 2 * (oculta1 + oculta2 + classes);

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
// GPU: treino de um batch em um único bloco
//
// As etapas que eram kernels separados na v3.2 viram funções do mesmo kernel,
// separadas por barreiras (__syncthreads). Cada etapa distribui seus elementos
// entre as threads do bloco, de blockDim.x em blockDim.x, então o tamanho do
// bloco é só um parâmetro. Os vetores do batch ficam na memória compartilhada:
// o valor k da amostra s fica em [s * largura + k]. Cada soma é feita na mesma
// ordem do barenn.cpp; as diferenças de arredondamento vêm da contração em FMA
// feita pelo nvcc.
// ---------------------------------------------------------------------------

// Forward de uma camada. Um elemento por neurônio de uma amostra:
// idx = s * out_features + i.
__device__ void forward_camada(const float* entrada, const float* W, const float* b,
                               float* saida, int n, int in_features, int out_features,
                               bool final) {
    for (int idx = threadIdx.x; idx < n * out_features; idx += blockDim.x) {
        const int s = idx / out_features;
        const int i = idx % out_features;
        const float* x = entrada + s * in_features;
        const float* w = W + i * in_features; // linha i
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
// positiva, o que equivale a z > 0.
__device__ void propagar_delta(const float* delta, const float* W, const float* ativacao_anterior,
                               float* delta_anterior, int n, int in_features, int out_features) {
    for (int idx = threadIdx.x; idx < n * in_features; idx += blockDim.x) {
        const int s = idx / in_features;
        const int j = idx % in_features;
        const float* d = delta + s * out_features;
        float grad = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            grad += d[i] * W[i * in_features + j];
        }
        delta_anterior[idx] = grad * (ativacao_anterior[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// Atualização dos parâmetros. Um elemento por peso e um por viés: cada um soma
// as correções das n amostras do batch, na ordem das amostras, e aplica a
// média. Nenhuma outra thread escreve no mesmo parâmetro.
__device__ void atualizar_camada(float* W, float* b, const float* delta, const float* entrada,
                                 int n, int in_features, int out_features,
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
            W[idx] += soma * inv;
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

// Treina um batch de n amostras em um único bloco. Os pesos ficam na memória
// global; a entrada, as ativações e os deltas do batch, na compartilhada.
__global__ void __launch_bounds__(threads_por_bloco)
treinar_batch_kernel(const float* entrada, const int* rotulos,
                     float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
                     float* perda_amostra, int n, float taxa, float inv) {
    extern __shared__ float compartilhada[];
    float* x = compartilhada;      // n × entradas
    float* a1 = x + n * entradas;  // n × oculta1
    float* a2 = a1 + n * oculta1;  // n × oculta2
    float* a3 = a2 + n * oculta2;  // n × classes (escores)
    float* d1 = a3 + n * classes;  // n × oculta1
    float* d2 = d1 + n * oculta1;  // n × oculta2
    float* d3 = d2 + n * oculta2;  // n × classes

    // A entrada do batch é lida da memória global uma vez e reaproveitada no
    // forward e na atualização da primeira camada.
    for (int k = threadIdx.x; k < n * entradas; k += blockDim.x) {
        x[k] = entrada[k];
    }
    __syncthreads();

    forward_camada(x, W1, b1, a1, n, entradas, oculta1, false);
    __syncthreads();
    forward_camada(a1, W2, b2, a2, n, oculta1, oculta2, false);
    __syncthreads();
    forward_camada(a2, W3, b3, a3, n, oculta2, classes, true);
    __syncthreads();
    perda_saida(a3, rotulos, d3, perda_amostra, n);
    __syncthreads();

    // Propaga o delta da última camada até a primeira, com os pesos ainda sem
    // a atualização deste batch.
    propagar_delta(d3, W3, a2, d2, n, oculta2, classes);
    __syncthreads();
    propagar_delta(d2, W2, a1, d1, n, oculta1, oculta2);
    __syncthreads();

    // Uma atualização por batch, com a média das correções das n amostras.
    // As três camadas não dependem umas das outras: sem barreira entre elas.
    atualizar_camada(W3, b3, d3, a2, n, oculta2, classes, taxa, inv);
    atualizar_camada(W2, b2, d2, a1, n, oculta1, oculta2, taxa, inv);
    atualizar_camada(W1, b1, d1, x, n, entradas, oculta1, taxa, inv);
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

        constexpr int epocas = 51;
        const int N = static_cast<int>(treino.tamanho());
        const int max_batch = min(batch_size, N);

        // Os vetores de um batch precisam caber na memória compartilhada de um bloco.
        const size_t bytes_por_amostra = Barenn::floats_por_amostra * sizeof(float);
        int compartilhada_por_bloco = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&compartilhada_por_bloco,
                                          cudaDevAttrMaxSharedMemoryPerBlock, 0));
        const int max_batch_gpu = static_cast<int>(compartilhada_por_bloco / bytes_por_amostra);
        if (max_batch > max_batch_gpu) {
            throw runtime_error("BATCH_SIZE deve ser <= " + to_string(max_batch_gpu) +
                                " nesta GPU: o batch precisa caber na memória compartilhada de um bloco");
        }

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
        CUDA_CHECK(cudaFuncGetAttributes(&atributos, Barenn::treinar_batch_kernel));

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
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (int inicio_batch = 0; inicio_batch < N; inicio_batch += batch_size) {
                // O último batch pode ser menor que batch_size.
                const int n = min(batch_size, N - inicio_batch);
                const float inv = 1.0f / n;
                const float* x = imagens + size_t(inicio_batch) * Barenn::entradas;

                // Um kernel de um bloco treina o batch inteiro.
                Barenn::treinar_batch_kernel<<<1, Barenn::threads_por_bloco, n * bytes_por_amostra>>>(
                    x, rotulos + inicio_batch, g1.W, g1.b, g2.W, g2.b, g3.W, g3.b,
                    perda_amostra + size_t(epoca) * N + inicio_batch, n, taxa, inv);
            }
        }
        // Os lançamentos são assíncronos: espera a GPU terminar antes de parar.
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
