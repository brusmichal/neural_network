from mnist8 import MnistSet8
from mnist28 import DEFAULT_MNIST28
from neural_network import Network
import matplotlib.pyplot as plt

mnist = MnistSet8()
training_set = mnist.get_training_set()
validating_set = mnist.get_validating_set()
testing_set = mnist.get_testing_set()

def simulation(network_layers, epochs, mini_batch_size, learning_rate):
    network = Network(network_layers)
    accuracy_history = []
    mse_history = []
    print('Starting simulation...')
    for i in range(epochs):
        network.sgd(training_set, mini_batch_size, learning_rate)
        valid_goods = network.evaluate(validating_set)
        valid_accuracy = valid_goods / len(validating_set) * 100
        goods = network.evaluate(testing_set)
        accuracy = goods / len(testing_set) * 100
        accuracy_history.append(accuracy)
        mse = network.mse(testing_set)
        mse_history.append(mse)
        info = f'Epoch {i+1}/{epochs}'
        print(f'{info:<15} -> valid_accuracy={valid_accuracy:.2f}%, accuracy={accuracy:.2f}%, mse={mse:.5f}')

    plt.figure()
    plt.plot(range(1, epochs+1), accuracy_history)
    plt.xlabel('Numer epoki')
    plt.ylabel('Procent trafień')
    plt.title('Trafienia sieci neuronowej w zależności od epoki')
    plt.grid(True)
    plt.figure()
    plt.plot(range(1, epochs+1), mse_history)
    plt.xlabel('Numer epoki')
    plt.ylabel('Błąd średniokwadratowy')
    plt.title('Błąd średniokwadratowy w zależności od epoki')
    plt.grid(True)
    plt.show()

simulation([64, 16, 16, 10], 50, 10, 3.0)
