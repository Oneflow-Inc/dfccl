# DFCCL
Deadlock Free Collective Communication Library.


## Compiling

```shell
make -j<n>
```
> Using `-gencode=arch=compute_86,code=sm_86` for `NVCC_GENCODE` by default. Set the `NVCC_GENCODE` environment variable when needed.

## Cite

```
@inproceedings{pan2025dfccl,
  title={Comprehensive Deadlock Prevention for GPU Collective Communication},
  author={Lichen Pan, Juncheng Liu, Yongquan Fu, Jinhui Yuan, Rongkai Zhang, Pengze Li, Zhen Xiao},
  booktitle={EuroSys'25},
  year={2025}
}
```

ISBN: 979-8-4007-1196-1/25/03

DOI: https://doi.org/10.1145/3689031.3717466

https://github.com/Oneflow-Inc/dfccl/
The commit ID corresponding to the code used in the paper is **66260c9**.
