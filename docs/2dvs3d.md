## Using U-Net in 2D and 3D

From the configuration file, the U-Net model can be configured to operate in either 2D or 3D by simply adjusting the `dimension` parameter. This flexibility allows you to choose the appropriate model architecture based on the nature of your data.

### Configuration Overview

The `dimension` parameter is located under the `data` section of your configuration file. By default, this is set to `2`, which configures the U-Net model to work with 2D data. To switch to a 3D U-Net model, you only need to change this value to `3`.

### Example Configuration for 2D U-Net

```yaml
data:
  dimension: 2
  # other parameters
  patch_size: 512
  # additional settings
```

### Example Configuration for 3D U-Net

```yaml
data:
  dimension: 3
  # other parameters
  patch_size: 64  # Typical size for 3D patches
  # additional settings
```

### How It Works

- **2D U-Net**: When `dimension: 2`, the model operates on 2D slices of your data, making it ideal for tasks like image segmentation where each input is a 2D image.
- **3D U-Net**: When `dimension: 3`, the model processes 3D volumes, which is useful for 3D medical imaging or volumetric data where spatial context across multiple planes is important.

### Patch Size Considerations

When switching to 3D, it’s often necessary to adjust the `patch_size` parameter due to the increased computational complexity. For 3D data, a smaller patch size is typically used to keep memory usage manageable.

### Summary

By changing the `dimension` parameter in your configuration, you can easily switch between 2D and 3D U-Net models. This simple adjustment allows the same pipeline to handle both 2D and 3D data with minimal changes, making your setup versatile and adaptable to various types of data.

