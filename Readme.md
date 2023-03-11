### 运行方式
```shell
D:/Users/86176/anaconda3/envs/PyTorch/python.exe DeepKmeans.py --data_file ./data/sample_151509.h5
```



### DeepKmeans

Function  `DeepKmeans` which takes an argument `args`. The purpose of this function is to implement a clustering method using a combination of Variational Autoencoder (VAE) and K-means algorithm.

- The first line of the function sets the random seed for repeatability.
- The second line of the function loads data from a file specified by the `args.data_file` parameter.
- The `x` and `labels` variables are assigned the values of the 'X' and 'Y' keys, respectively, from the data file.
- The data file is closed after reading.
- The next block of code preprocesses the data using the `AnnData` class from the `scanpy` library.
- The `adata` variable is assigned an `AnnData` object created from the `x` variable.
- The 'Group' attribute of the `obs` property of the `adata` object is set to the `labels` variable.
- The 'DCA_split' attribute of the `obs` property is set to 'train' and its type is set to 'category'.
- The next block normalizes the data using the `sc.pp.normalize_per_cell()` function and applies other preprocessing steps such as filtering genes and cells, creating a copy of the raw data, and log-transforming the data.
- Next, the VAE model is initialized with the specified input dimension, z_dim, encodeLayer, decodeLayer, sigma and device.
- The model is then trained using the trainAE function on the preprocessed data.
- The code then estimates the number of clusters by applying the Louvain algorithm on the autoencoder latent representations
- The code then uses the estimated number of clusters and the initial cluster centers obtained by louvain algorithm to fit the K-means model
- The final latent representations and predicted labels are saved to the specified files



### Main

是使用参数解析库 `argparse` 设置命令行参数，并使用这些参数对 `DeepKmeans` 和 `plot_res` 函数进行调用。具体而言，使用者可以通过命令行设置以下参数：

- `n_neighbors`：整数，默认值为`76`；
- `resolution`：浮点数，默认值为`0.8`, `Louvain` 的参数；
- `batch_size`：整数，默认值为`256`；
- `data_file`：字符串，默认为 `'./data/sample_151509.h5'`；
- `epoch_num`：整数，默认值为`24`；
- `sigma`：浮点数，默认值为`0.8`, `VAE` 的参数；
- `final_latent_file`：字符串，默认值为`'final_latent_file.txt'`，表示存储`z`的路径；
- `predict_label_file`：字符串，默认值为`'pred_labels.txt'`，表示存储聚类结果的路径；
- `device`：字符串，默认值为`'cuda:0'`。

设置完参数后，程序会打印参数的值并对 `DeepKmeans` 和 `plot_res` 函数进行调用。
