from fastapi import FastAPI, File, UploadFile
from source.services.detection import detection
from fastapi.responses import StreamingResponse
from source.utils.generic_utils import get_bytestream_from_image, image_input
from PIL import ImageDraw, Image
import io


app = FastAPI()


@app.post("/detection/")
async def table_detection(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_tables = detection(decoded_image_rgb, [width, height], image_name)
    return detected_tables


@app.post("/result_visualisation/")
async def result_visualisation(file: UploadFile = File(...)):
    image_content = await file.read()
    decoded_image_rgb = image_input(image_content)
    image_name = file.filename
    width, height, _ = decoded_image_rgb.shape
    detected_tables = detection(decoded_image_rgb, [width, height], image_name)
    image_bytestream = get_bytestream_from_image(decoded_image_rgb)
    image = Image.open(image_bytestream)
    draw = ImageDraw.Draw(image, "RGBA")
    for i in detected_tables:
        bbox = i.bbox
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline='red', width=3)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    result = output.getvalue()
    return StreamingResponse(io.BytesIO(result), media_type="image/png")
