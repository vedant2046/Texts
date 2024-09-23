# from TTS.api import TTS
# tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda:2")
# tts.voice_conversion_to_file(source_wav="_t1_source.wav", target_wav="_t1_target.wav", file_path="_t1.wav")


#try this 

from TTS.api import TTS

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)
tts.model.to("cuda:2")  
tts.voice_conversion_to_file(source_wav="_t1_source.wav", target_wav="_t1_target.wav", file_path="_t1.wav")
