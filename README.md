# NeuralNetwork
Библиотека для обучения полносвязных нейросетей. Пример использования в папке **example**.

# Сборка
git clone https://github.com/ArsenyPogosov/NeuralNetwork.git \
mkdir build && cd $_ \
cmake ../NeuralNetwork \
ccmake . //Тут можно включить флаги NEURAL_NETWORK_BUILD_TESTS и NEURAL_NETWORK_BUILD_EXAMPLE \
cmake --build .
