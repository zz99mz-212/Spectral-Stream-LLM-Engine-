from spectralstream.format.core import (
    SSF_MAGIC,
    SSF_HEADER_SIZE,
    SSF_FOOTER_SIZE,
    SSF_PAGE_SIZE,
    SSF_REDUNDANT_HEADER_OFFSET,
    SSFVersion,
    TensorDType,
    _align_up,
    _sha256,
    _format_size,
    _LOSSY_TO_METHOD,
    _LEGACY_COMPRESSION_MAP,
)
from spectralstream.format.compression import (
    _import_method_registry,
    _method_id_to_name,
    _name_to_method_id,
    _compress_via_engine,
    _decompress_via_engine,
    _get_engine_method,
    _ENGINE_METHODS,
    _ENGINE_LOCK,
)
from spectralstream.format.header import SSFHeader
from spectralstream.format.index import (
    TensorIndexEntry,
    LegacyTensorIndexEntry,
    TensorIndex,
)
from spectralstream.format.writer import SSFWriter
from spectralstream.format.reader import SSFReader
from spectralstream.format.converter import SSFConverter
from spectralstream.format.conversion_report import (
    ConversionReport,
    LayerReport,
    TensorReport,
)

try:
    from spectralstream.format.sst_format import (
        SSTv3Writer,
        SSTv3Reader,
        SSTReader,
        SST_MAGIC,
        SST_VERSION,
    )
except ImportError:
    SSTv3Writer = None  # type: ignore
    SSTv3Reader = None  # type: ignore
    SSTReader = None  # type: ignore
    SST_MAGIC = None  # type: ignore
    SST_VERSION = None  # type: ignore
from spectralstream.format.model_converter import ModelConverter

try:
    from spectralstream.format.sscx_format import (
        SSCXWriter,
        SSCXReader,
        SSCXHeader,
        SSCXLayerEntry,
        SSCXTensorEntry,
        SSCXFooter,
        SSCX_MAGIC,
        SSCX_VERSION,
        SSCX_HEADER_SIZE,
        SSCX_PAGE_SIZE,
    )
except ImportError:
    (
        SSCXWriter,
        SSCXReader,
        SSCXHeader,
        SSCXLayerEntry,
        SSCXTensorEntry,
        SSCXFooter,
        SSCX_MAGIC,
        SSCX_VERSION,
        SSCX_HEADER_SIZE,
        SSCX_PAGE_SIZE,
    ) = (None,) * 10

__all__ = [
    "SSF_MAGIC",
    "SSF_HEADER_SIZE",
    "SSF_FOOTER_SIZE",
    "SSF_PAGE_SIZE",
    "SSF_REDUNDANT_HEADER_OFFSET",
    "SSFVersion",
    "TensorDType",
    "_align_up",
    "_sha256",
    "_format_size",
    "_import_method_registry",
    "_method_id_to_name",
    "_name_to_method_id",
    "_compress_via_engine",
    "_decompress_via_engine",
    "_get_engine_method",
    "SSFHeader",
    "TensorIndexEntry",
    "LegacyTensorIndexEntry",
    "TensorIndex",
    "SSFWriter",
    "SSFReader",
    "SSFConverter",
    "ConversionReport",
    "LayerReport",
    "TensorReport",
    "SSTv3Writer",
    "SSTv3Reader",
    "SSTReader",
    "SST_MAGIC",
    "SST_VERSION",
    "ModelConverter",
    "SSCXWriter",
    "SSCXReader",
    "SSCXHeader",
    "SSCXLayerEntry",
    "SSCXTensorEntry",
    "SSCXFooter",
    "SSCX_MAGIC",
    "SSCX_VERSION",
    "SSCX_HEADER_SIZE",
    "SSCX_PAGE_SIZE",
]
