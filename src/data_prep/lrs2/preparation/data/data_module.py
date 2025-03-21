#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
import os
import subprocess


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True):
        self.modality = modality
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=convert_gray)

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            if not landmarks:
                landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            return video

    def load_audio(self, data_filename):
        if data_filename.endswith('.mp4'):
            # Convert mp4 to wav first
            wav_path = data_filename.replace('.mp4', '.wav')
            if not os.path.exists(wav_path):
                command = [
                    'ffmpeg',
                    '-i', data_filename,  # input
                    '-vn',  # no video
                    '-acodec', 'pcm_s16le',  # audio codec
                    '-ar', '16000',  # sample rate
                    '-ac', '1',  # mono
                    wav_path  # output
                ]
                subprocess.run(command, capture_output=True)
            
            # Now load the wav file
            waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
            return waveform, sample_rate
        else:
            # Original code for non-mp4 files
            waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
            return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
