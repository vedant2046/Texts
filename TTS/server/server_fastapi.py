"""
这是coqui-tts的fastapi接口
pip install fastapi uvicorn python-multipart
"""
import io
import json
import os
import logging
import shutil
import uuid
import traceback
from pathlib import Path
from threading import Lock
from typing import Union, List, Optional
from pydantic import Field, BaseModel
from fastapi import FastAPI, Query, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

app = FastAPI()
lock = Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

# 将正在使用的模型更新为指定的已发布模型。
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict, None]:
    """
    该函数功能是将一个uri样式的wave文件路径或GST（Guided Style Tokens）
    字典转换为相应的格式。如果输入是一个字符串类型的路径，且该路径指向一个以".wav"结尾的文件，
    则直接返回该路径。如果输入是一个字符串类型的JSON表示的GST字典，则将其解析为字典格式并返回。
    如果输入为空字符串或既不是文件路径也不是GST字典的字符串，则返回None。
    Args:
        style_wav (str): uri
    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav 是位于服务器上的.wav文件
        style_wav = json.loads(style_wav)
        return style_wav  # style_wav 是带有 {token1_id ： token1_weigth， ...} 的 GST 字典
    return None


@app.get("/list_models", summary="获取模型列表")
async def get_list_models():
    """
    获取所有可用模型列表
    """
    tts_models = manager.list_tts_models()
    vocoder_models = manager.list_vocoder_models()
    return {"tts_models": tts_models, "vocoder_models": vocoder_models}


@app.get("/load_model", summary="加载模型")
async def load_models(
        model_name: str = Query(description="模型名称"),
        vocoder_name: str = Query(default=None, description="vocoder名称，一般不用填"),
        config_path: str = Query(default=None, description="配置文件路径，一般不用填，只有`tts_models/multilingual/multi-dataset/xtts_v2`模型需要填 `/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json`")
):
    """
    加载模型
    """
    global synthesizer

    model_path, tts_config_path, model_item = manager.download_model(model_name)
    vocoder_name = model_item["default_vocoder"] if vocoder_name is None else vocoder_name

    if vocoder_name is not None:
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    if config_path is not None:
        tts_config_path = config_path

    # 加载模型
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=tts_config_path,
        tts_speakers_file="",
        tts_languages_file="",
        vocoder_checkpoint="",
        vocoder_config="",
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=os.getenv("CUDA", False)
    )
    # 是否使用多发言人模式
    speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)
    # 是否使用多语言模式
    language_manager = getattr(synthesizer.tts_model, "language_manager", None)

    return {
        "message": "模型加载成功",
        "speaker_manager": speaker_manager,
        "language_manager": language_manager
    }


class TTSRequest(BaseModel):
    """
    语音合成请求
    """
    text: Optional[str] = Field(description="需要转换的文本")
    speaker_idx: Optional[str] = Field(default=None, description="说话人 id")
    language_idx: Optional[str] = Field(default=None, description="语言 id")
    speed: Optional[float] = Field(default=1.0, description="生成音频的速度。默认为 1.0。（如果远低于 1.0，可能会产生伪影）"),
    split_sentences: Optional[bool] = Field(default=True, description="将输入文本拆分为句子")
    # Todo: 下面参数还没看懂实质上的作用；暂时不添加到接口中
    # style_wav: Optional[str] = Field(default=None, description="GST的样式波形")
    # style_text: Optional[str] = Field(default=None, description="Capacitron 的 style_wav 转录")
    # reference_wav: Optional[str] = Field(default=None, description="用于语音转换的参考波形")
    # reference_speaker_name: Optional[str] = Field(default=None, description="参考波形的扬声器 ID")

@app.get("/tts", summary="语音合成")
def tts(
        text: Optional[str] = Query(description="需要转换的文本"),
        speaker_idx: Optional[str] = Query(default=None, description="说话人"),
        language_idx: Optional[str] = Query(default=None, description="语种；如：en"),
        speed: Optional[float] = Query(default=1.0, description="生成音频的速度。默认为 1.0。（如果远低于 1.0，可能会产生伪影）"),
        split_sentences: Optional[bool] = Query(default=True, description="将输入文本拆分为句子")
):
    """
    语音合成
    """
    try:
        with lock:
            # 使用异常处理来增加健壮性
            try:
                wavs = synthesizer.tts(
                    text=text,
                    speaker_name=speaker_idx,
                    language_name=language_idx,
                    split_sentences=split_sentences,
                    speed=speed
                )
                out = io.BytesIO()
                synthesizer.save_wav(wavs, out)
                # synthesizer.save_wav(wavs, "output_tts.wav")
            except AttributeError as e:
                logger.error(f"语音合成失败: {e}")
                raise HTTPException(status_code=501, detail="未加载TTS模型")
            except Exception:
                logger.error(f"语音合成失败: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="语音合成失败")
    except Exception as e:
        logger.error(f"处理请求时出错: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    headers = {
        'Content-Disposition': 'inline; filename="output.wav"'
    }
    return StreamingResponse(out, media_type="audio/wav", headers=headers)
    # return FileResponse("output_tts.wav", media_type="audio/wav", filename="output.wav")


@app.post("/clone_tts", summary="语音克隆并合成")
def clone_tts(
        text: Optional[str] = Form(description="需要转换的文本"),
        speaker_idx: Optional[str] = Form(default=None, description="说话人"),
        language_idx: Optional[str] = Form(default=None, description="语种；如：en"),
        speed: Optional[float] = Form(default=1.0, description="生成音频的速度。默认为 1.0。（如果远低于 1.0，可能会产生伪影）"),
        split_sentences: Optional[bool] = Form(default=True, description="将输入文本拆分为句子"),
        speaker_wav: List[UploadFile] = File(..., description="需要克隆的说话人音频wav文件；支持单个或多个文件；单个音频需要大于3s；wav文件数量和种类决定了克隆的效果")
):
    """
    语音克隆并合成
    """
    try:
        with lock:
            # 创建缓存目录
            cache_dir = "wav_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # 保存上传的文件并获取文件路径
            speaker_wav_paths = []
            if speaker_wav is not None:
                for file in speaker_wav:
                    file_id = str(uuid.uuid4())
                    file_path = os.path.join(cache_dir, f"{file_id}_{file.filename}")
                    with open(file_path, "wb") as f:
                        shutil.copyfileobj(file.file, f)
                    speaker_wav_paths.append(file_path)

            logger.info(f" > Model input: {text}")
            logger.info(f" > Language Idx: {language_idx}")

            # 使用异常处理来增加健壮性
            try:
                wavs = synthesizer.tts(
                    text=text,
                    language_name=language_idx,
                    speaker_wav=speaker_wav_paths,
                    split_sentences=split_sentences,
                    speed=speed,
                    speaker_name=speaker_idx
                )
                out = io.BytesIO()
                synthesizer.save_wav(wavs, out)
            except AttributeError as e:
                logger.error(f"语音合成失败: {e}")
                raise HTTPException(status_code=501, detail="未加载TTS模型")
            except Exception:
                logger.error(f"语音合成失败: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="语音合成失败")
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理缓存文件
        for file_path in speaker_wav_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        # 如果缓存目录为空，则删除
        if not os.listdir(cache_dir):
            os.rmdir(cache_dir)
    return StreamingResponse(out, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5002, log_level="debug")
