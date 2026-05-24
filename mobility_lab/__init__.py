# scikit-mobility 1.2.x imports cascaded_union, removed in Shapely 2.0 (use unary_union).
import shapely.ops as _shapely_ops

if not hasattr(_shapely_ops, "cascaded_union"):
    _shapely_ops.cascaded_union = _shapely_ops.unary_union
