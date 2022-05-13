# Artificial Neural Network

Implementation of a multi-layer perceptron with backpropagation and gradient optimization. Trained and tested with data: [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist).

## Programs

### `mnist8.py` and `mnist28.py`

Those program print digits from MNIST set to the terminal:

```bash
python mnist8.py [image-index]
python mnist28.py [image-index]
```

Default value of `[image-index]` is 0.

Both programs require RGB terminal support:

```bash
# Test for RGB terminal
printf "\x1b[38;2;40;177;249mTRUECOLOR\x1b[0m\n"
```
