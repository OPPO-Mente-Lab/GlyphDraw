from pytorch_lightning import LightningDataModule
from typing import Optional
import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from custom_dataset_en import ImageEmbeddingDataset, collate_custom, expand_urls
import webdataset as wds
from prefetch_generator import BackgroundGenerator


def get_consume_samples(data_model: LightningDataModule) -> int:
    if hasattr(data_model.trainer.lightning_module, 'consumed_samples'):
        consumed_samples = data_model.trainer.lightning_module.consumed_samples
        print('get consumed samples from model: {}'.format(consumed_samples))
    else:
        world_size = data_model.trainer.world_size
        consumed_samples = max(0, data_model.trainer.global_step - 1) * \
            data_model.hparams.train_batchsize * world_size * data_model.trainer.accumulate_grad_batches
        print('calculate consumed samples: {}'.format(consumed_samples))
    return consumed_samples


class UniversalDataModule(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--dataloader_workers', default=2, type=int)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--val_batchsize', default=16, type=int)
        parser.add_argument('--test_batchsize', default=16, type=int)
        parser.add_argument('--datasets_name', type=str, default=None)
        parser.add_argument('--train_datasets_field', type=str, default='train')
        parser.add_argument('--val_datasets_field', type=str, default='validation')
        parser.add_argument('--test_datasets_field', type=str, default='test')
        parser.add_argument('--train_file', type=str, default=None)
        parser.add_argument('--val_file', type=str, default=None)
        parser.add_argument('--test_file', type=str, default=None)
        parser.add_argument('--raw_file_type', type=str, default='json')
        parser.add_argument('--sampler_type', type=str,
                            choices=['single',
                                     'random'],
                            default='random')
        return parent_args

    def __init__(
        self,
        tokenizer,
        collate_fn,
        args,
        datasets=None,
        **kwargs,
    ):
        super().__init__()
        # 如果不传入datasets的名字，则可以在对象外部替换内部的datasets为模型需要的
        if datasets is not None:
            self.datasets = datasets
        elif args.datasets_name is not None:
            from fengshen.data.fs_datasets import load_dataset
            print('---------begin to load datasets {}'.format(args.datasets_name))
            self.datasets = load_dataset(
                args.datasets_name, num_proc=args.num_workers)
            print('---------ending load datasets {}'.format(args.datasets_name))
        else:
            print('---------begin to load datasets from local file')
            from datasets import load_dataset
            self.datasets = load_dataset(args.raw_file_type,
                                         data_files={
                                             args.train_datasets_field: args.train_file,
                                             args.val_datasets_field: args.val_file,
                                             args.test_datasets_field: args.test_file})
            print('---------end to load datasets from local file')

        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.save_hyperparameters(args)

    def get_custom_sampler(self, ds):
        from universal_sampler import PretrainingRandomSampler
        from universal_sampler import PretrainingSampler
        world_size = self.trainer.world_size
        consumed_samples = get_consume_samples(self)
        # use the user default sampler
        if self.hparams.sampler_type == 'random':
            return PretrainingRandomSampler(
                total_samples=len(ds),
                # consumed_samples cal by global steps
                consumed_samples=consumed_samples,
                micro_batch_size=self.hparams.train_batchsize,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=world_size,
                epoch=self.trainer.current_epoch,
            )
        elif self.hparams.sampler_type == 'single':
            return PretrainingSampler(
                total_samples=len(ds),
                # consumed_samples cal by global steps
                consumed_samples=consumed_samples,
                micro_batch_size=self.hparams.train_batchsize,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=world_size,
            )
        else:
            raise Exception('Unknown sampler type: {}'.format(self.hparams.sampler_type))

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def train_dataloader(self):
        ds = self.datasets[self.hparams.train_datasets_field]

        collate_fn = self.collate_fn
        if collate_fn is None and hasattr(ds, 'collater'):
            collate_fn = ds.collater

        if self.hparams.replace_sampler_ddp is False:
            return DataLoader(
                ds,
                batch_sampler=self.get_custom_sampler(ds),
                num_workers=self.hparams.dataloader_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        return DataLoader(
            ds,
            batch_size=self.hparams.train_batchsize,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = self.datasets[self.hparams.val_datasets_field]
        collate_fn = self.collate_fn
        if collate_fn is None and hasattr(ds, 'collater'):
            collate_fn = ds.collater

        return DataLoader(
            ds,
            batch_size=self.hparams.val_batchsize,
            shuffle=False,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=collate_fn,
            sampler=DistributedSampler(
                ds, shuffle=False),
            pin_memory=True,
        )

        # return DataLoader(
        #     ds, shuffle=False, batch_size=self.hparams.val_batchsize, pin_memory=False, collate_fn=collate_fn,
        # )

    def test_dataloader(self):
        ds = self.datasets[self.hparams.test_datasets_field]

        collate_fn = self.collate_fn
        if collate_fn is None and hasattr(ds, 'collater'):
            collate_fn = ds.collater

        return DataLoader(
            ds,
            batch_size=self.hparams.test_batchsize,
            shuffle=False,
            num_workers=self.hparams.dataloader_workers,
            collate_fn=collate_fn,
            sampler=DistributedSampler(
                ds, shuffle=False),
            pin_memory=True,
        )


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataModuleCustom(LightningDataModule):
    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Universal DataModule')
        parser.add_argument('--webdataset_base_urls', type=str, nargs="+")
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--batch_size', default=16, type=int)
        # parser.add_argument('--start_shard', default=0, type=int)
        # parser.add_argument('--end_shard', default=1000, type=int)
        parser.add_argument('--shard_width', default=5, type=int)
        parser.add_argument('--hr_size', default=-1, type=int)
        parser.add_argument('--train_split', default=1.0, type=float)
        parser.add_argument('--val_split', default=0.0, type=float)
        parser.add_argument('--test_split', default=0.0, type=float)
        parser.add_argument('--shuffle_train', default=False, action="store_true")
        parser.add_argument('--resample_train', default=False, action="store_true")
        parser.add_argument('--shuffle_num', default=None, type=int)
        parser.add_argument('--test_prompts', type=str, default="./test_prompts.txt")
        parser.add_argument('--test_repeat', default=1, type=int)

        parser.add_argument(
            "--resolution", type=int, default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", default=True,
            help="Whether to center crop images before resizing to resolution"
        )
        return parent_args

    def __init__(
        self,
        args,
        tokenizer,
        collate_fn = None,
        use_worker_init_fn=None,
    ):
        super().__init__()
        # self.available_shards = list(range(args.start_shard, args.end_shard + 1))
        # if splits is None:
        #     splits = []
        splits = {
            'train': args.train_split,
            'val': args.val_split,
            'test': args.test_split,
        }
        self.webdataset_base_urls = args.webdataset_base_urls
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.shuffle_train = args.shuffle_train
        self.resample_train = args.resample_train
        self.shard_width = args.shard_width
        self.hr_size = args.hr_size
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_num = args.shuffle_num
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn if collate_fn is not None else collate_custom
        self.center_crop = args.center_crop
        self.resolution = args.resolution

        self.train_prop = self.val_prop = self.test_prop = 0
        self.datasets = {}
        if splits['train'] > 0:
            self.train_prop = splits['train']
            self.train_dataloader = self._train_dataloader
            self.datasets['train'] = None
        if splits['val'] > 0:
            self.val_prop = splits['val']
            self.val_dataloader = self._val_dataloader
            self.datasets['val'] = None
        if splits['test'] > 0:
            self.test_prop = splits['test']
            self.test_dataloader = self._test_dataloader
            self.datasets['test'] = None
        
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        assert self.train_prop + self.test_prop + self.val_prop == 1
        # num_train = round(self.train_prop*len(self.available_shards))
        # num_test = round(self.test_prop*len(self.available_shards))
        # num_val = len(self.available_shards) - num_train - num_test
        # assert num_train + num_test + num_val == len(self.available_shards), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(self.available_shards)}"
        # train_split, test_split, val_split = random_split(self.available_shards, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)
        # self.train_urls = [self.webdataset_base_url.format(str(shard).zfill(self.shard_width)) for shard in train_split]
        # self.test_urls = [self.webdataset_base_url.format(str(shard).zfill(self.shard_width)) for shard in test_split]
        # self.val_urls = [self.webdataset_base_url.format(str(shard).zfill(self.shard_width)) for shard in val_split]

        all_urls = []
        for url in self.webdataset_base_urls:
            all_urls += expand_urls(url)
        num_train = round(self.train_prop*len(all_urls))
        num_test = round(self.test_prop*len(all_urls))
        num_val = len(all_urls) - num_train - num_test
        assert num_train + num_test + num_val == len(all_urls), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(all_urls)}"
        self.train_urls, self.test_urls, self.val_urls = random_split(all_urls, [num_train, num_test, num_val])  # , generator=torch.Generator().manual_seed(self.seed)
        
    def setup(self, stage=None):
        if 'train' in self.datasets:
            self.datasets['train'] = ImageEmbeddingDataset(
                self.train_urls,
                self.tokenizer,
                shuffle_shards=self.shuffle_train,
                resample=self.resample_train,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )
            if self.shuffle_num is not None and self.shuffle_num > 0:
                self.datasets['train'].shuffle(self.shuffle_num)
        if 'val' in self.datasets:
            self.datasets['val'] = ImageEmbeddingDataset(
                self.val_urls,
                self.tokenizer,
                shuffle_shards=False,
                resample=False,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )
        if 'test' in self.datasets:
            self.datasets['test'] = ImageEmbeddingDataset(
                self.test_urls,
                self.tokenizer,
                shuffle_shards=False,
                resample=False,
                hr_size=self.hr_size,
                handler=wds.handlers.warn_and_continue,
                center_crop=self.center_crop,
                size=self.resolution,
            )

    def _train_dataloader(self):
        # return self.create_dataloader(self.train_urls, shuffle=self.shuffle_train, resample=self.resample_train)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['train'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

    def _val_dataloader(self, shuffle=False):
        # return self.create_dataloader(self.val_urls, shuffle=False)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['val'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )

    def _test_dataloader(self, shuffle=False):
        # return self.create_dataloader(self.test_urls, shuffle=False)
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets['test'],
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn,
        )
