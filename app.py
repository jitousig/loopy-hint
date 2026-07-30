import cv2
import numpy as np
import streamlit as st

from pipeline import ExtractionError, hint_from_image

st.set_page_config(page_title="Loopy Hint", page_icon="🔁")
st.title("Loopy Hint")
st.caption(
    "Upload a screenshot of a pentagonal-grid Loopy puzzle (SGT Puzzles). "
    "You'll get a mistake check and one next move — not the full solution."
)

up = st.file_uploader("Puzzle screenshot", type=["png", "jpg", "jpeg"])

if up is not None:
    data = np.frombuffer(up.read(), np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        st.error("That file didn't decode as an image. Upload a PNG or JPEG.")
    else:
        with st.spinner("Reading the grid and solving..."):
            try:
                msg, out = hint_from_image(im)
            except ExtractionError as ex:
                st.error(str(ex))
                st.stop()
            except Exception:
                st.error(
                    "Something unexpected went wrong reading this screenshot. "
                    "Try a full, unzoomed screenshot straight from the app."
                )
                st.stop()
        st.success(msg) if "mistake" not in msg.lower() or "no mistakes" in msg.lower() \
            else st.warning(msg)
        # crop to the drawn puzzle area for display
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        content = np.where((gray < 150).any(axis=1))[0]
        y0, y1 = (content.min() - 20, content.max() + 20) if len(content) else (0, out.shape[0])
        st.image(cv2.cvtColor(out[max(0, y0):y1], cv2.COLOR_BGR2RGB),
                 caption="Highlighted edge(s): blue = draw in, red = cross out / fix",
                 use_container_width=True)

st.divider()
st.caption(
    "Blue highlight: draw that edge. Red highlight: cross it out, or fix the "
    "flagged mistake. The hint engine only ever reveals one forced step."
)
