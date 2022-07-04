
from argparse import ArgumentError
import collections
from collections import OrderedDict
import copy
from typing import Any, Callable, Dict, List, Optional, Union
import glog as log
import torch.distributed as dist
import torch

class Sample(OrderedDict):
    """Sample represent some arbitrary data. All datasets in MMSC must
    return an object of type ``Sample``.

    Args:
        init_dict (Dict): Dictionary to init ``Sample`` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())




class SampleList(OrderedDict):
    """``SampleList`` is used to collate a list of ``Sample`` into a batch during batch
    preparation. It can be thought of as a merger of list of Dicts into a single Dict.

    If ``Sample`` contains an attribute 'text' of size (2) and there are 10 samples in
    list, the returned ``SampleList`` will have an attribute 'text' which is a tensor
    of size (10, 2).

    Args:
        samples (type): List of ``Sample`` from which the ``SampleList``
                        will be created.

    Usage::

        >>> sample_list = [
                Sample({"text": torch.tensor(2)}),
                Sample({"text": torch.tensor(2)})
            ]
        >>> sample_list.text
        torch.tensor([2, 2])
    """

    _TENSOR_FIELD_ = "_tensor_field"

    def __init__(self, samples=None):
        super().__init__(self)
        if samples is None:
            samples = []

        if len(samples) == 0:
            return

        if self._check_and_load_dict(samples):
            return
        # If passed sample list was in form of key, value pairs of tuples
        # return after loading these
        if self._check_and_load_tuple(samples):
            return

        fields = samples[0].keys()

        for field in fields:
            if isinstance(samples[0][field], torch.Tensor):
                size = (len(samples), *samples[0][field].size())
                self[field] = samples[0][field].new_empty(size)
                if self._get_tensor_field() is None:
                    self._set_tensor_field(field)
            else:
                self[field] = [None for _ in range(len(samples))]

            for idx, sample in enumerate(samples):
                # it should be a tensor but not a 0-d tensor
                if (
                    isinstance(sample[field], torch.Tensor)
                    and len(sample[field].size()) != 0
                    and sample[field].size(0) != samples[0][field].size(0)
                ):
                    raise AssertionError(
                        "Fields for all samples must be equally sized. "
                        "{} is of different sizes".format(field)
                    )

                self[field][idx] = self._get_data_copy(sample[field])

            if isinstance(samples[0][field], collections.abc.Mapping):
                self[field] = SampleList(self[field])

    def _check_and_load_tuple(self, samples):
        if isinstance(samples[0], (tuple, list)) and isinstance(samples[0][0], str):
            for kv_pair in samples:
                self.add_field(kv_pair[0], kv_pair[1])
            return True
        else:
            return False

    def _check_and_load_dict(self, samples):
        if isinstance(samples, collections.abc.Mapping):
            for key, value in samples.items():
                self.add_field(key, value)
            return True
        else:
            return False

    def _fix_sample_type(self, samples):
        if not isinstance(samples[0], Sample):
            proper_samples = []
            for sample in samples:
                proper_samples.append(Sample(sample))
            samples = proper_samples
        return samples

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(
                "Key {} not found in the SampleList. "
                "Valid choices are {}".format(key, self.fields())
            )
        fields = self.keys()

        if key in fields:
            return self[key]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][key]

        return sample

    def get_device(self):
        field_tensor = self._get_tensor_field()
        assert (
            field_tensor is not None
        ), f"No tensor field in sample list, available keys: {self.fields()}"
        return self[field_tensor].device

    def get_item_list(self, key):
        """Get ``SampleList`` of only one particular attribute that is present
        in the ``SampleList``.

        Args:
            key (str): Attribute whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute value of the key
            which was passed.

        """
        sample = self[key]

        return SampleList([sample])

    def copy(self):
        """Get a copy of the current SampleList

        Returns:
            SampleList: Copy of current SampleList.

        """
        sample_list = SampleList()

        fields = self.fields()

        for field in fields:
            sample_list.add_field(field, self[field])

        return sample_list

    def fields(self):
        """Get current attributes/fields registered under the SampleList.

        Returns:
            List[str]: list of attributes of the SampleList.

        """
        return list(self.keys())

    def get_fields(self, fields):
        """Get a new ``SampleList`` generated from the current ``SampleList``
        but contains only the attributes passed in `fields` argument

        Args:
            fields (List[str]): Attributes whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute values of the fields
            which were passed.

        """
        current_fields = self.fields()

        return_list = SampleList()

        for field in fields:
            if field not in current_fields:
                raise AttributeError(
                    "{} not present in SampleList. "
                    "Valid choices are {}".format(field, current_fields)
                )
            return_list.add_field(field, self[field])

        return return_list

    def get_field(self, field):
        """Get value of a particular attribute

        Args:
            field (str): Attribute whose value is to be returned.
        """
        return self[field]

    def _get_data_copy(self, data):
        # if isinstance(data, torch.Tensor):
        #     copy_ = data.clone()
        # else:
        #     copy_ = deepcopy(data)
        # return copy_
        return data

    def _get_tensor_field(self):
        return self.__dict__.get(SampleList._TENSOR_FIELD_, None)

    def _set_tensor_field(self, value):
        self.__dict__[SampleList._TENSOR_FIELD_] = value

    def get_batch_size(self):
        """Get batch size of the current ``SampleList``. There must be a tensor
        be a tensor present inside sample list to use this function.
        Returns:
            int: Size of the batch in ``SampleList``.

        """
        tensor_field = self._get_tensor_field()
        assert tensor_field is not None, "There is no tensor yet in SampleList"

        return self[tensor_field].size(0)

    def add_field(self, field, data):
        """Add an attribute ``field`` with value ``data`` to the SampleList

        Args:
            field (str): Key under which the data will be added.
            data (object): Data to be added, can be a ``torch.Tensor``, ``list``
                         or ``Sample``
        """
        fields = self.fields()
        tensor_field = self._get_tensor_field()

        if (
            len(fields) != 0
            and isinstance(data, torch.Tensor)
            and len(data.size()) != 0
            and tensor_field is not None
            and data.size(0) != self[tensor_field].size(0)
        ):
            raise AssertionError(
                "A tensor field to be added must "
                "have same size as existing tensor "
                "fields in SampleList. "
                "Passed size: {}, Required size: {}".format(
                    len(data), len(self[tensor_field])
                )
            )

        if isinstance(data, collections.abc.Mapping):
            self[field] = SampleList(data)
        else:
            self[field] = self._get_data_copy(data)

            if isinstance(self[field], torch.Tensor) and tensor_field is None:
                self._set_tensor_field(field)

    def to(self, device, non_blocking=True):
        """Similar to ``.to`` function on a `torch.Tensor`. Moves all of the
        tensors present inside the ``SampleList`` to a particular device. If an
        attribute's value is not a tensor, it is ignored and kept as it is.

        Args:
            device (str|torch.device): Device on which the ``SampleList`` should
                                       moved.
            non_blocking (bool): Whether the move should be non_blocking. Default: True

        Returns:
            SampleList: a SampleList moved to the ``device``.
        """
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], "to"):
                sample_list[field] = sample_list[field].to(
                    device, non_blocking=non_blocking
                )

        return sample_list

    def pin_memory(self):
        """In custom batch object, we need to define pin_memory function so that
        PyTorch can actually apply pinning. This function just individually pins
        all of the tensor fields
        """
        fields = self.keys()

        for field in fields:
            if hasattr(self[field], "pin_memory"):
                # This will also handle nested sample list recursively
                self[field] = self[field].pin_memory()

        return self

    def detach(self):
        fields = self.keys()

        for field in fields:
            self[field] = detach_tensor(self[field])

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts a sample list to dict, this is useful for TorchScript and for
        other internal API unification efforts.

        Returns:
            Dict[str, Any]: A dict representation of current sample list
        """
        sample_dict = {}
        fields = self.keys()

        for field in fields:
            # Handle nested sample list recursively
            if hasattr(self[field], "to_dict"):
                sample_dict[field] = self[field].to_dict()
            else:
                sample_dict[field] = self[field]

        return sample_dict


class Report(OrderedDict):
    def __init__(
        self, batch: SampleList = None, model_output: Dict[str, Any] = None, *args
    ):
        super().__init__(self)
        if batch is None:
            return
        if model_output is None:
            model_output = {}
        if self._check_and_load_tuple(batch):
            return

        all_args = [batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.abc.Mapping):
                raise TypeError(
                    "Argument {:d}, {} must be of instance of "
                    "collections.abc.Mapping".format(idx, arg)
                )

        self.batch_size = batch.get_batch_size()
        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "in previous arguments to report"
                    )
                    log.warn(log)
                self[key] = item

    def get_batch_size(self) -> int:
        return self.batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def _check_and_load_tuple(self, batch):
        if isinstance(batch, collections.abc.Mapping):
            return False

        if isinstance(batch[0], (tuple, list)) and isinstance(batch[0][0], str):
            for kv_pair in batch:
                self[kv_pair[0]] = kv_pair[1]
            return True
        else:
            return False

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self) -> List[str]:
        return list(self.keys())

    def apply_fn(self, fn: Callable, fields: Optional[List[str]] = None):
        """Applies a function `fn` on all items in a report. Can apply to specific
        fields if `fields` parameter is passed

        Args:
            fn (Callable): A callable to called on each item in report
            fields (List[str], optional): Use to apply on specific fields.
                Defaults to None.

        Returns:
            Report: Update report after apply fn
        """
        for key in self.keys():
            if fields is not None and isinstance(fields, (list, tuple)):
                if key not in fields:
                    continue
            self[key] = fn(self[key])
            if isinstance(self[key], collections.MutableSequence):
                for idx, item in enumerate(self[key]):
                    self[key][idx] = fn(item)
            elif isinstance(self[key], dict):
                for subkey in self[key].keys():
                    self[key][subkey] = fn(self[key][subkey])
        return self

    def detach(self) -> "Report":
        """Similar to tensor.detach, detach all items in a report from their graphs.
        This is useful in clearing up memory sometimes.

        Returns:
            Report: Detached report is returned back.
        """
        return self.apply_fn(detach_tensor)

    def to(
        self,
        device: Union[torch.device, str],
        non_blocking: bool = True,
        fields: Optional[List[str]] = None,
    ):
        """Move report to a specific device defined 'device' parameter.
        This is similar to how one moves a tensor or sample_list to a device

        Args:
            device (torch.device): Device can be str defining device or torch.device
            non_blocking (bool, optional): Whether transfer should be non_blocking.
                Defaults to True.
            fields (List[str], optional): Use this is you only want to move some
                specific fields to the device instead of full report. Defaults to None.

        Raises:
            TypeError: If device type is not correct

        Returns:
            Report: Updated report is returned back
        """
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        def fn(x):
            if hasattr(x, "to"):
                x = x.to(device, non_blocking=non_blocking)
            return x

        return self.apply_fn(fn, fields)

    def accumulate_tensor_fields_and_loss(
        self, report: "Report", field_list: List[str]
    ):
        self._accumulate_tensor_fields(report, field_list)
        self._accumulate_loss(report)
    
    def _accumulate_tensor_fields(
        self, report: "Report", field_list: List[str]
    ):
        for key in field_list:
            if key == "__prediction_report__":
                continue
            if key not in self.keys():
                log.warn(
                    f"{key} not found in report. Metrics calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat((self[key], report[key]), dim=0)
            
            if isinstance(self[key], list):
                self[key].extend(report[key])
        
        self.batch_size = self.batch_size + report.batch_size
    
    def _accumulate_loss(self, report: "Report"):
        for key, value in report.losses.items():
            if key not in self.losses.keys():
                log.warn(
                    f"{key} not found in report. Loss calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self.losses[key], torch.Tensor):
                self.losses[key] += value

    def _gather_fileds(self, field_list: List[str]):
        assert dist.is_initialized()
        rank = dist.get_rank()
        for key in field_list:
            assert key in self.keys()
            
            if isinstance(self[key], torch.Tensor):
                tensor_list = [torch.zeros_like(self[key]) for _ in range(dist.get_world_size())]
                # if rank == 0:
                #     dist.gather(self[key], gather_list=tensor_list, dst=0)
                # else:
                #     dist.gather(self[key], dst=0)
                dist.all_gather(tensor_list, self[key])
                # try:
                #     dist.all_gather(tensor_list, self[key])
                # except:
                #     log.info('all gather fail')
                #     log.info(key)
                #     log.info(self[key])
                #     raise ArgumentError(None, 'all gather fail error')

                if rank == 0:
                    self[key] = torch.cat(tensor_list, dim=0)
            
            if isinstance(self[key], list):
                gather_list = [None for _ in range(dist.get_world_size())]
                # if rank == 0:
                #     dist.gather_object(self[key], gather_list=gather_list, dst=0)
                # else:
                #     dist.gather_object(self[key], dst=0)
                dist.all_gather_object(gather_list, self[key])
                
                if rank==0:
                    output_list = []
                    for li in gather_list:
                        output_list.extend(li)
                    self[key]=output_list
        
        self.batch_size = len(self[key])

    def _eliminate_redundant_index(self, field_list: List[str]):
        assert 'index' in self.keys()
        assert type(self['index'])==list and type(self['index'][0])==list

        for key in field_list:
            assert key in self.keys()
            assert len(self['index'])==len(self[key])
            accumulate_list = []
            index_clear = []
            for i in range(len(self['index'])):
                if self['index'][i][0] in index_clear:
                    continue
                index_clear.append(self['index'][i][0])

                accumulate_list.append(self[key][i])
            
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.stack(accumulate_list, dim=0)
            else:
                self[key] = accumulate_list

        self['index_clear'] = index_clear
        self.batch_size = len(index_clear)

    def copy(self) -> "Report":
        """Get a copy of the current Report

        Returns:
            Report: Copy of current Report.

        """
        report = Report()

        fields = self.fields()

        for field in fields:
            report[field] = copy.deepcopy(self[field])

        return report

def detach_tensor(tensor: Any) -> Any:
    """Detaches any element passed which has a `.detach` function defined.
    Currently, in MMSC can be SampleList, Report or a tensor.

    Args:
        tensor (Any): Item to be detached

    Returns:
        Any: Detached element
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    return tensor

def gather(tensor, tensor_list=None, root=0):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
  
    rank = dist.get_rank()
    if rank == root:
        assert(tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list)
    else:
        dist.gather(tensor, dst=root)