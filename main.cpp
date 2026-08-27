#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;



class Tensor {
    double* memoria;
    vector<size_t> dimension;
    size_t tamanio_total;
    public:
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor ( const vector < size_t >& shape ,
const vector < double >& values ) : dimension(shape) {
        tamanio_total = 1;
        for (size_t dim : shape) {
            tamanio_total *= dim;
        }
        memoria = new double[tamanio_total];

        for (size_t i = 0; i < tamanio_total; ++i) {
            memoria[i] = values[i];
        }
    }

    ~Tensor() {
        delete[] memoria;
    }


    void mostrar_info() const {
        cout << "Tensor de " << dimension.size() << "D creado con "
                  << tamanio_total << " elementos en memoria." << endl;
    }

    static Tensor zeros( const vector<size_t>& shape ) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        vector<double> vals(total, 0.0);
        return Tensor(shape, vals);
    }

    static Tensor ones(const vector<size_t>& shape ) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        vector<double> vals(total, 1.0);
        return Tensor(shape, vals);
    }

    static Tensor arange(double start, double end) {
        vector<double> vals;
        for (double v= start; v < end; v++) {
            vals.push_back(v);
        }
        return Tensor({vals.size()}, vals);
    }
    static Tensor random(const vector<size_t> shape,double min_val = 0.0, double max_val=1.0){
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(min_val, max_val);

    vector<double> vals(total);
    for (size_t i=0; i<total; i++) {
        vals[i] = dis(gen);
    }
        return Tensor(shape, vals);
    }

    Tensor(const Tensor& other) : dimension(other.dimension), tamanio_total(other.tamanio_total) {
        memoria = new double[tamanio_total];
        for (size_t i = 0; i < tamanio_total; ++i) {
            memoria[i] = other.memoria[i];
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete[] memoria;

            dimension = other.dimension;
            tamanio_total = other.tamanio_total;
            memoria = new double[tamanio_total];
            for (size_t i = 0; i < tamanio_total; ++i) {
                memoria[i] = other.memoria[i];
            }
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept
    : memoria(other.memoria), dimension(move(other.dimension)), tamanio_total(other.tamanio_total) {
        other.memoria = nullptr;
        other.tamanio_total = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] memoria;

            memoria = other.memoria;
            dimension = move(other.dimension);
            tamanio_total = other.tamanio_total;

            other.memoria = nullptr;
            other.tamanio_total = 0;
        }
        return *this;
    }

    Tensor operator+(const Tensor& other) const {
        // Caso 1: Dimensiones idénticas
        if (dimension == other.dimension) {
            vector<double> res_vals(tamanio_total);
            for (size_t i = 0; i < tamanio_total; ++i) {
                res_vals[i] = memoria[i] + other.memoria[i];
            }
            return Tensor(dimension, res_vals);
        }
        // Caso 2: Suma de Bias (1 x N) a una matriz (M x N)
        if (dimension.size() == 2 && other.dimension.size() == 2 &&
            other.dimension[0] == 1 && dimension[1] == other.dimension[1]) {
            size_t M = dimension[0];
            size_t N = dimension[1];
            vector<double> res_vals(tamanio_total);
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res_vals[i * N + j] = memoria[i * N + j] + other.memoria[j];
                }
            }
            return Tensor(dimension, res_vals);
            }
        throw invalid_argument("Dimensiones incompatibles para la suma.");
    }
    Tensor operator-(const Tensor& other) const {
        if (dimension != other.dimension) {
            throw invalid_argument("Dimensiones incompatibles para la resta.");
        }
        vector<double> res_vals(tamanio_total);
        for (size_t i = 0; i < tamanio_total; ++i) {
            res_vals[i] = memoria[i] - other.memoria[i];
        }
        return Tensor(dimension, res_vals);
    }
    Tensor operator*(const Tensor& other) const {
        if (dimension != other.dimension) {
            throw invalid_argument("Dimensiones incompatibles para la multiplicacion.");
        }
        vector<double> res_vals(tamanio_total);
        for (size_t i = 0; i < tamanio_total; ++i) {
            res_vals[i] = memoria[i] * other.memoria[i];
        }
        return Tensor(dimension, res_vals);
    }
    Tensor operator*(double scalar) const {
        vector<double> res_vals(tamanio_total);
        for (size_t i = 0; i < tamanio_total; ++i) {
            res_vals[i] = memoria[i] * scalar;
        }
        return Tensor(dimension, res_vals);
    }

    Tensor view(const vector<size_t>& new_shape) const {
        if (new_shape.size() > 3) {
            throw invalid_argument("El numero de dimensiones no puede exceder 3.");
        }

        size_t new_total = 1;
        for (size_t dim : new_shape) {
            new_total *= dim;
        }

        if (new_total != tamanio_total) {
            throw invalid_argument("El numero total de elementos debe mantenerse constante en view.");
        }

        vector<double> vals(memoria, memoria + tamanio_total);
        return Tensor(new_shape, vals);
    }

    Tensor unsqueeze(size_t axis) const {
        if (dimension.size() >= 3) {
            throw invalid_argument("El tensor ya alcanza el limite maximo de 3 dimensiones.");
        }
        if (axis > dimension.size()) {
            throw invalid_argument("El eje especificado esta fuera de rango.");
        }

        vector<size_t> new_shape = dimension;
        new_shape.insert(new_shape.begin() + axis, 1); // Inserta '1' en la posición 'axis'

        vector<double> vals(memoria, memoria + tamanio_total);
        return Tensor(new_shape, vals);
    }

    static Tensor concat(const vector<Tensor>& tensors, size_t axis) {
        if (tensors.empty()) {
            throw invalid_argument("La lista de tensores no puede estar vacia.");
        }

        size_t num_dims = tensors[0].dimension.size();
        if (axis >= num_dims) {
            throw invalid_argument("El eje especificado esta fuera de rango.");
        }

        vector<size_t> new_shape = tensors[0].dimension;
        size_t total_axis_dim = 0;

        for (const auto& t : tensors) {
            if (t.dimension.size() != num_dims) {
                throw invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones.");
            }
            for (size_t d = 0; d < num_dims; ++d) {
                if (d != axis && t.dimension[d] != tensors[0].dimension[d]) {
                    throw invalid_argument("Las dimensiones deben coincidir en todos los ejes excepto en el eje de concatenacion.");
                }
            }
            total_axis_dim += t.dimension[axis];
        }

        new_shape[axis] = total_axis_dim;

        size_t outer_size = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer_size *= new_shape[i];
        }

        size_t inner_size = 1;
        for (size_t i = axis + 1; i < num_dims; ++i) {
            inner_size *= new_shape[i];
        }

        size_t new_total_size = outer_size * total_axis_dim * inner_size;
        vector<double> res_vals(new_total_size);

        size_t write_idx = 0;
        for (size_t o = 0; o < outer_size; ++o) {
            for (const auto& t : tensors) {
                size_t copy_count = t.dimension[axis] * inner_size;
                size_t read_idx = o * copy_count;
                for (size_t i = 0; i < copy_count; ++i) {
                    res_vals[write_idx++] = t.memoria[read_idx + i];
                }
            }
        }
        return Tensor(new_shape, res_vals);
    }

    Tensor relu() const {
        vector<double> res_vals(tamanio_total);
        for (size_t i = 0; i < tamanio_total; ++i) {
            res_vals[i] = memoria[i] > 0.0 ? memoria[i] : 0.0;
        }
        return Tensor(dimension, res_vals);
    }

    Tensor sigmoid() const {
        vector<double> res_vals(tamanio_total);
        for (size_t i = 0; i < tamanio_total; ++i) {
            res_vals[i] = 1.0 / (1.0 + std::exp(-memoria[i]));
        }
        return Tensor(dimension, res_vals);
    }
};

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.dimension != b.dimension) {
        throw invalid_argument("Las dimensiones deben ser exactamente iguales para el producto punto.");
    }

    double suma = 0.0;
    for (size_t i = 0; i < a.tamanio_total; ++i) {
        suma += a.memoria[i] * b.memoria[i];
    }

    return Tensor({1}, {suma});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dimension.size() != 2 || b.dimension.size() != 2) {
        throw invalid_argument("matmul solo esta implementado para matrices bidimensionales (2D).");
    }
    if (a.dimension[1] != b.dimension[0]) {
        throw invalid_argument("Dimensiones incompatibles para matmul: las columnas de A deben coincidir con las filas de B.");
    }

    size_t M = a.dimension[0];
    size_t K = a.dimension[1];
    size_t N = b.dimension[1];

    vector<double> res_vals(M * N, 0.0);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                res_vals[i * N + j] += a.memoria[i * K + k] * b.memoria[k * N + j];
            }
        }
    }

    return Tensor({M, N}, res_vals);
}


int main() {
    cout << "=== SIMULACION DE RED NEURONAL ===" << endl;

    //(1000 x 20 x 20)
    Tensor x = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    cout << "Paso 1 - Entrada: "; x.mostrar_info();

    //1000 x 400
    Tensor x_flat = x.view({1000, 400});
    cout << "Paso 2 - view: "; x_flat.mostrar_info();

    //(400 x 100)
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor h1 = matmul(x_flat, W1);
    cout << "Paso 3 - matmul W1: "; h1.mostrar_info();

    //(1 x 100)
    Tensor b1 = Tensor::zeros({1, 100});
    Tensor h1_bias = h1 + b1;
    cout << "Paso 4 - Suma b1: "; h1_bias.mostrar_info();

    //ReLU
    Tensor a1 = h1_bias.relu();
    cout << "Paso 5 - ReLU: "; a1.mostrar_info();

    //(100 x 10)
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor h2 = matmul(a1, W2);
    cout << "Paso 6 - matmul W2: "; h2.mostrar_info();

    //(1 x 10)
    Tensor b2 = Tensor::zeros({1, 10});
    Tensor h2_bias = h2 + b2;
    cout << "Paso 7 - Suma b2: "; h2_bias.mostrar_info();

    //Activacion Sigmoid
    Tensor out = h2_bias.sigmoid();
    cout << "Paso 8 - Sigmoid (Salida Final): "; out.mostrar_info();

    return 0;
}