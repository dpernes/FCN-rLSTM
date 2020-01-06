# FCN-rLSTM (Vehicle Counting)

Implementation of the model FCN-rLSTM, as described in the paper: 
Zhang et al., "FCN-rLSTM: Deep spatio-temporal neural networks for vehicle counting in city cameras", *ICCV 2017*.
https://arxiv.org/abs/1707.09476

This code was not written by any of the authors of the original work, so small differences may exist. Use it at your own risk.
If you use the model implemented here in a publication, please do not forget to cite the paper:
```
@inproceedings{zhang2017fcn,
  title={Fcn-rlstm: Deep spatio-temporal neural networks for vehicle counting in city cameras},
  author={Zhang, Shanghang and Wu, Guanhang and Costeira, Joao P and Moura, Jos{\'e} MF},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3667--3676},
  year={2017}
}
```

## Dataset
All experiments use the [TRANCOS dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/).
If you make use of this data, please cite the following reference in any publications:
```
@inproceedings{TRANCOSdataset_IbPRIA2015,
  title={Extremely Overlapping Vehicle Counting},
  author={Ricardo Guerrero-Gómez-Olmedo, Beatriz Torre-Jiménez, Roberto López-Sastre, Saturnino Maldonado Bascón, and Daniel Oñoro-Rubio},
  booktitle={Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA)},
  year={2015}
}
```
Camera annotations in `cam_annotations.txt` are my own and they are required for the experiments with sequential data. A handy script for manual camera annotation is provided in `annotate_cams.py`.
