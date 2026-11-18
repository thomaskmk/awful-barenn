#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Estado de uma camada durante o treino: os rascunhos de uma amostra e os
// acumuladores do batch. A camada guarda só os parâmetros; cada thread usa
// o próprio Estado de cada camada.
struct alignas(64) Estado {
    // Dados do último forward usados para calcular os gradientes.
    const vector<float>* last_input = nullptr; // aponta para a entrada, sem copiar
    vector<float> last_z;
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;
    // Acumuladores do mini-batch: correções já com o sinal e a taxa.
    vector<vector<float>> grad_w;
    vector<float> grad_b;
};

class DenseLayer {
    int in_features;
    int out_features;
    bool inicial;
    bool final;
    // weights[i][j] liga a entrada j ao neurônio de saída i.
    vector<vector<float>> weights;
    vector<float> bias;

public:
    DenseLayer(int input_size, int output_size, bool is_inicial = false, bool is_final = false)
        : in_features(input_size), out_features(output_size), inicial(is_inicial), final(is_final),
          weights(output_size, vector<float>(input_size)), bias(output_size, 0) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (auto& row : weights) {
            for (float& weight : row) {
                weight = distribution(gen);
            }
        }
    }

    // Cria um Estado com os tamanhos desta camada.
    Estado NovoEstado() const {
        Estado e;
        e.last_z.resize(out_features);
        e.out.resize(out_features);
        e.delta.resize(out_features);
        e.grad_in.resize(in_features);
        e.grad_w.assign(out_features, vector<float>(in_features, 0.0f));
        e.grad_b.assign(out_features, 0.0f);
        return e;
    }

    // const: o forward só lê os parâmetros e escreve no Estado.
    const vector<float>& Forward(const vector<float>& input, Estado& e) const {
        e.last_input = &input;
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            // Laço escalar de propósito: o #pragma omp simd reduction gerou mais
            // instruções e deixou o forward mais lento (ver issue #8).
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i][j];
            }
            e.last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            e.out[i] = final ? sum : max(0.0f, sum);
        }
        return e.out;
    }

    // const: o backward só acumula correções no Estado; os pesos mudam no Atualizar.
    const vector<float>& Backward(const vector<float>& grad_out, float learning_rate, Estado& e) const {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            e.delta[i] = final ? grad_out[i]
                               : grad_out[i] * (e.last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente do peso é delta vezes a entrada; o do viés é o próprio delta.
        // A camada de entrada não calcula grad_in: ninguém o recebe.
        const vector<float>& x = *e.last_input;
        if (!inicial) {
            fill(e.grad_in.begin(), e.grad_in.end(), 0.0f);
            for (int i = 0; i < out_features; ++i) {
                const float d = e.delta[i];
                for (int j = 0; j < in_features; ++j) {
                    e.grad_in[j] += d * weights[i][j];
                    e.grad_w[i][j] -= learning_rate * (d * x[j]);
                }
                e.grad_b[i] -= learning_rate * d;
            }
        } else {
            for (int i = 0; i < out_features; ++i) {
                const float d = e.delta[i];
                for (int j = 0; j < in_features; ++j) {
                    e.grad_w[i][j] -= learning_rate * (d * x[j]);
                }
                e.grad_b[i] -= learning_rate * d;
            }
        }

        return e.grad_in;
    }

    // Soma as correções acumuladas por todas as threads, em ordem fixa de thread
    // (resultado reproduzível), aplica a média e zera os acumuladores.
    // n é o número de amostras do batch (o último batch pode ser menor).
    void Atualizar(vector<Estado>& estados, size_t n) {
        const float inv = 1.0f / n;
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                float soma = estados[0].grad_w[i][j];
                estados[0].grad_w[i][j] = 0.0f;
                for (size_t t = 1; t < estados.size(); ++t) {
                    soma += estados[t].grad_w[i][j];
                    estados[t].grad_w[i][j] = 0.0f;
                }
                weights[i][j] += soma * inv;
            }
            float soma_b = estados[0].grad_b[i];
            estados[0].grad_b[i] = 0.0f;
            for (size_t t = 1; t < estados.size(); ++t) {
                soma_b += estados[t].grad_b[i];
                estados[t].grad_b[i] = 0.0f;
            }
            bias[i] += soma_b * inv;
        }
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(std::move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(const DenseLayer& l1, const DenseLayer& l2, const DenseLayer& l3,
             const vector<Amostra>& teste) {
    // Só forward: um Estado por camada, sem uso dos acumuladores.
    Estado e1 = l1.NovoEstado();
    Estado e2 = l2.NovoEstado();
    Estado e3 = l3.NovoEstado();
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

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem, e1), e2), e3);
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

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
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

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
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
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
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32, /*inicial=*/true);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, /*inicial=*/false, /*final=*/true);
        array<double, epocas> perdas{};

        // Estado de treino por thread e por camada (rascunhos e acumuladores),
        // criado uma vez. Cada thread aloca o próprio estado: a memória fica
        // separada entre as threads e no processador de quem a usa.
        const int nthreads = omp_get_max_threads();
        vector<Barenn::Estado> e1(nthreads), e2(nthreads), e3(nthreads);
        vector<vector<float>> gradientes(nthreads);
        #pragma omp parallel
        {
            const int t = omp_get_thread_num();
            e1[t] = l1.NovoEstado();
            e2[t] = l2.NovoEstado();
            e3[t] = l3.NovoEstado();
            gradientes[t].assign(Barenn::classes, 0.0f);
        }

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t inicio_batch = 0; inicio_batch < treino.size(); inicio_batch += batch_size) {
                // O último batch pode ser menor que batch_size.
                size_t fim_batch = min(inicio_batch + batch_size, treino.size());

                // As amostras do batch são independentes: cada thread processa
                // as suas com o próprio estado. static: trabalho igual por amostra
                // e a mesma divisão em toda execução.
                #pragma omp parallel for schedule(static) reduction(+:perda_total)
                for (size_t s = inicio_batch; s < fim_batch; ++s) {
                    const int t = omp_get_thread_num();
                    const auto& amostra = treino[s];
                    const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem, e1[t]), e2[t]), e3[t]);
                    perda_total += Barenn::perda(escores, amostra.rotulo, &gradientes[t]);

                    // Propaga o gradiente da última camada até a primeira.
                    // Cada Backward acumula as correções no estado da thread, sem alterar os pesos.
                    const auto& grad3 = l3.Backward(gradientes[t], taxa, e3[t]);
                    const auto& grad2 = l2.Backward(grad3, taxa, e2[t]);
                    l1.Backward(grad2, taxa, e1[t]);
                }

                // Uma atualização por batch, com a média das correções de todas as threads.
                const size_t n = fim_batch - inicio_batch;
                l1.Atualizar(e1, n);
                l2.Atualizar(e2, n);
                l3.Atualizar(e3, n);
            }
            perdas[epoca] = perda_total / treino.size();
        }

        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
