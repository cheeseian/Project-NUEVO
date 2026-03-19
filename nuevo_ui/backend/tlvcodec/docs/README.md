# TLV Codec (Python)

Python encoder/decoder for the NUEVO TLV serial protocol.
Matches the C implementation in `firmware/arduino/src/lib/tlvcodec.c`.

## Usage

```python
from tlvcodec.src.encoder import encode_frame
from tlvcodec.src.decoder import Decoder

# Encode a TLV frame
frame_bytes = encode_frame(tlv_type, payload_bytes)

# Decode incoming bytes
decoder = Decoder()
for tlv_type, payload_bytes in decoder.feed(raw_bytes):
    handle(tlv_type, payload_bytes)
```

## Location

This package lives at `nuevo_ui/backend/tlvcodec/` and is installed as a local dependency alongside `nuevo_bridge` (`pip install -e .`).

It was originally part of `ros2_ws/src/tlvcodec/` and was moved here so `nuevo_bridge` can run without ROS2.
