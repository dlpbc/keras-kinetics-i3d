# keras-kinetics-i3d
Keras implementation (including pretrained weights) of Inflated 3D Inception architecture reported in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

See original implementation by the authors. [repository](https://github.com/deepmind/kinetics-i3d)

# Usage
```
python eval.py

or

[For help]
python eval.py -h
```

# Requirements
- Keras
- Keras Backend: Tensorflow (tested) or Theano (not tested) or CNTK (not tested)
- h5py

# License
- All code in this repository are licensed under the MIT license as specified by the LICENSE file.
- The i3d (rgb and flow) weights were ported from the ones released [Deepmind](https://deepmind.com) in this [repository](https://github.com/deepmind/kinetics-i3d) under [Apache-2.0 License](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)
