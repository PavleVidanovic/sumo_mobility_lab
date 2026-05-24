"""
Folium animacija: SVG ikonice vozila i sinhronizovani Leaflet TimeDimension sloj.
"""

from __future__ import annotations

import base64

import numpy as np
from branca.element import MacroElement
from folium.elements import JSCSSMixin
from folium.folium import Map
from folium.template import Template
from folium.utilities import remove_empty

VEHICLE_ICON_CLASSES = (
    "passenger",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "pedestrian",
)

_ICON_URI_CACHE: dict[str, str] = {}

_SVG_BY_CLASS: dict[str, str] = {
    "truck": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24">'
        '<rect x="4" y="2" width="16" height="18" rx="2" fill="{fill}" stroke="#1a1a1a"/>'
        '<rect x="4" y="2" width="7" height="8" fill="#88b3e8" opacity="0.85"/>'
        '<rect x="2" y="10" width="3" height="5" rx="1" fill="#333"/>'
        '<rect x="19" y="10" width="3" height="5" rx="1" fill="#333"/>'
        "</svg>"
    ),
    "bus": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="22" viewBox="0 0 28 24">'
        '<rect x="3" y="4" width="22" height="14" rx="2" fill="{fill}" stroke="#1a1a1a"/>'
        '<rect x="5" y="6" width="18" height="5" fill="#88b3e8" opacity="0.9"/>'
        '<rect x="2" y="9" width="2" height="6" fill="#333"/>'
        '<rect x="24" y="9" width="2" height="6" fill="#333"/>'
        "</svg>"
    ),
    "bicycle": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">'
        '<circle cx="6" cy="17" r="3" fill="none" stroke="#333" stroke-width="1.5"/>'
        '<circle cx="18" cy="17" r="3" fill="none" stroke="#333" stroke-width="1.5"/>'
        '<path d="M6 17 L12 8 L16 12 L18 17" stroke="{fill}" stroke-width="2.5" fill="none"/>'
        '<circle cx="12" cy="7" r="2" fill="{fill}"/>'
        "</svg>"
    ),
    "motorcycle": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="20" viewBox="0 0 24 24">'
        '<ellipse cx="7" cy="17" rx="3" ry="3" fill="#333"/>'
        '<ellipse cx="17" cy="17" rx="3" ry="3" fill="#333"/>'
        '<path d="M7 17 Q12 6 17 17" stroke="{fill}" stroke-width="3" fill="none"/>'
        "</svg>"
    ),
    "pedestrian": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="20" viewBox="0 0 20 24">'
        '<circle cx="10" cy="5" r="3" fill="{fill}" stroke="#1a1a1a"/>'
        '<line x1="10" y1="8" x2="10" y2="15" stroke="{fill}" stroke-width="2.5"/>'
        '<line x1="10" y1="11" x2="5" y2="14" stroke="{fill}" stroke-width="2"/>'
        '<line x1="10" y1="11" x2="15" y2="14" stroke="{fill}" stroke-width="2"/>'
        '<line x1="10" y1="15" x2="6" y2="22" stroke="{fill}" stroke-width="2"/>'
        '<line x1="10" y1="15" x2="14" y2="22" stroke="{fill}" stroke-width="2"/>'
        "</svg>"
    ),
    "passenger": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">'
        '<rect x="6" y="3" width="12" height="17" rx="3" fill="{fill}" stroke="#1a1a1a" stroke-width="1"/>'
        '<rect x="3" y="8" width="4" height="6" rx="1" fill="#333"/>'
        '<rect x="17" y="8" width="4" height="6" rx="1" fill="#333"/>'
        '<rect x="8" y="5" width="8" height="4" rx="1" fill="#88b3e8" opacity="0.9"/>'
        "</svg>"
    ),
}


def _svg_to_data_uri(svg: str) -> str:
    """
    Pakuje SVG string u data-URI za Folium marker ikonu.
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def vehicle_icon_data_uri(
    fill_hex: str = "#2563eb",
    vehicle_class: str = "passenger",
) -> str:
    """
    Top-down ikonica po tipu vozila (data-URI SVG).
    """
    cls = vehicle_class if vehicle_class in VEHICLE_ICON_CLASSES else "passenger"
    key = f"{cls}:{fill_hex.lower()}"
    if key in _ICON_URI_CACHE:
        return _ICON_URI_CACHE[key]
    template = _SVG_BY_CLASS.get(cls, _SVG_BY_CLASS["passenger"])
    uri = _svg_to_data_uri(template.format(fill=fill_hex))
    _ICON_URI_CACHE[key] = uri
    return uri


class SyncedTimeAnimation(JSCSSMixin, MacroElement):
    """
    Folium sloj: jedan Leaflet TimeDimension za ulice i vozila (sinhron play/klizač).

    Dva odvojena TimestampedGeoJson sloja bi se resetovali međusobno, ovde je samo jedan.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
            L.Control.TimeDimensionCustom = L.Control.TimeDimension.extend({
                _getDisplayDateFormat: function(date){
                    return moment(date).format("{{this.date_options}}");
                }
            });
            var map = {{this._parent.get_name()}};
            map.timeDimension = L.timeDimension({ period: {{ this.period|tojson }} });
            map.addControl(new L.Control.TimeDimensionCustom({{ this.options|tojavascript }}));

            function bindPopups(feature, layer) {
                if (feature.properties.popup) {
                    layer.bindPopup(feature.properties.popup);
                }
            }

            function addTimedGeoJson(data, durationVal, updateDim) {
                if (!data.features || data.features.length === 0) {
                    return null;
                }
                var gj = L.geoJson(data, {
                    pointToLayer: function (feature, latLng) {
                        if (feature.properties.icon === 'marker' && feature.properties.iconstyle) {
                            return L.marker(latLng, { icon: L.icon(feature.properties.iconstyle) });
                        }
                        if (feature.properties.icon === 'circle' && feature.properties.iconstyle) {
                            return L.circleMarker(latLng, feature.properties.iconstyle);
                        }
                        return L.marker(latLng);
                    },
                    style: function (feature) {
                        return feature.properties.style || {};
                    },
                    onEachFeature: bindPopups
                });
                return L.timeDimension.layer.geoJson(gj, {
                    updateTimeDimension: updateDim,
                    addlastPoint: false,
                    duration: durationVal
                }).addTo(map);
            }

            var edgeData = {{ this.edge_data|tojson }};
            var vehData = {{ this.vehicle_data|tojson }};
            var edgeHas = edgeData.features && edgeData.features.length > 0;
            var vehHas = vehData.features && vehData.features.length > 0;
            if (edgeHas) {
                addTimedGeoJson(edgeData, undefined, true);
            }
            if (vehHas) {
                addTimedGeoJson(vehData, {{ this.vehicle_duration }}, !edgeHas);
            }
        {% endmacro %}
        """)

    default_js = [
        (
            "jquery3.7.1",
            "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js",
        ),
        (
            "jqueryui1.10.2",
            "https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.10.2/jquery-ui.min.js",
        ),
        (
            "iso8601",
            "https://cdn.jsdelivr.net/npm/iso8601-js-period@0.2.1/iso8601.min.js",
        ),
        (
            "leaflet.timedimension",
            "https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.min.js",
        ),
        (
            "moment",
            "https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.18.1/moment.min.js",
        ),
    ]
    default_css = [
        (
            "leaflet.timedimension_css",
            "https://cdn.jsdelivr.net/npm/leaflet-timedimension@1.1.1/dist/leaflet.timedimension.control.css",
        ),
    ]

    def __init__(
        self,
        edge_geojson: dict,
        vehicle_geojson: dict,
        *,
        period: str,
        vehicle_duration: str,
        auto_play: bool = False,
        date_options: str = "YYYY-MM-DD HH:mm:ss",
        transition_ms: int = 300,
    ):
        """
        edge_geojson / vehicle_geojson: FeatureCollection po kadrovima.

        period: ISO8601 trajanje koraka (npr. PT10S). transition_ms: trajanje prelaza između kadrova.
        """
        super().__init__()
        self._name = "SyncedTimeAnimation"
        self.edge_data = edge_geojson
        self.vehicle_data = vehicle_geojson
        self.period = period
        self.date_options = date_options
        self.vehicle_duration = (
            "undefined" if not vehicle_duration else '"' + vehicle_duration + '"'
        )
        transition_ms = int(np.clip(transition_ms, 100, 1200))
        self.options = remove_empty(
            position="bottomleft",
            auto_play=bool(auto_play),
            loopButton=True,
            timeSliderDragUpdate=True,
            speedSlider=False,
            playerOptions={"transitionTime": transition_ms, "loop": False},
        )

    def render(self, **kwargs):
        """Ubacuje JS/CSS TimeDimension u Folium mapu."""
        super().render(**kwargs)
        assert isinstance(self._parent, Map)
