import h5py
import yaml

from unsat.Sampler import RectangularSampler


def extract_samples_from_hdf5(yaml_path, hdf5_path, target_path, size):
    faults = {}
    with open(yaml_path, 'r') as yaml_file:
        samples = yaml.safe_load(yaml_file)

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        with h5py.File(target_path, 'w') as faults_file:
            for sample in samples:
                sample_group = faults_file.create_group(sample['sample'])
                day = sample['day']
                day_group = sample_group.create_group(f"{day}")
                print(f"Processing sample: {sample['sample']} on day {day}")
                dataset = hdf5_file[sample['sample']]['data'][day - 1]
                labels = hdf5_file[sample['sample']]['labels'][day - 1]

                for issue in sample['issues']:
                    print(f"  Issue: {issue['issue']}")
                    for entry in issue['entries']:
                        z, x, y = entry['z'], entry['x'], entry['y']
                        single_fault_group = day_group.create_group(f"{x}-{y}-{z}")
                        single_fault_group.attrs['issue'] = issue['issue']

                        if z < dataset.shape[0]:  # Ensure z-index is within bounds
                            data_sampler = RectangularSampler(dataset[z], (y, x), size)
                            label_sampler = RectangularSampler(labels[z], (y, x), size)
                            if data_sampler.is_out():
                                print(
                                    f"    Point ({x}, {y}, {z}) is out of bounds for specified size."
                                )
                            else:
                                single_fault_group.create_dataset(
                                    'data', data=data_sampler.sample()
                                )
                                single_fault_group.create_dataset(
                                    'labels', data=label_sampler.sample()
                                )
                        else:
                            print(f"    Z-index {z} is out of bounds for the dataset.")

    return faults


yaml_path = 'faults.yaml'
hdf5_path = '../data/data.h5'
target_path = '../data/faults.h5'
faults = extract_samples_from_hdf5(yaml_path, hdf5_path, target_path, size=(25, 25))
