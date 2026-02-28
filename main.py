#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取与分类脚本

功能描述:
    1. 从音频文件提取20个音频特征 (文件级平均值)
    2. 加载本地 XGBoost 模型与 Scaler
    3. 输出男性/女性预测概率
    
输出:
    - 特征CSV文件 (用于记录)
    - 屏幕打印预测结果 (含特征权重分析)
"""
import joblib
import sys
import os
import argparse
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import warnings
import xgboost as xgb

# 忽略librosa和xgboost的警告
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """
    音频特征提取器类
    """
    
    def __init__(self, audio_file, sr=22050, hop_length=512):
        self.audio_file = audio_file
        self.sr = sr
        self.hop_length = hop_length
        
        # 加载音频文件
        print(f"[加载] {audio_file}...", end=" ", flush=True)
        try:
            self.y, self.sr = librosa.load(audio_file, sr=sr, mono=True)
            print(f"✓ ({len(self.y)} samples, {self.sr} Hz)")
        except Exception as e:
            print(f"✗\n错误: {e}")
            raise
        
        # 计算STFT
        print(f"[计算] STFT频谱...", end=" ", flush=True)
        try:
            self.D = librosa.stft(self.y, hop_length=hop_length)
            self.magnitude = np.abs(self.D)
            self.frequencies = librosa.fft_frequencies(sr=self.sr)
            print(f"✓ ({self.magnitude.shape[1]} frames)")
        except Exception as e:
            print(f"✗\n错误: {e}")
            raise
        
        # 提取基频(Fundamental Frequency)
        print(f"[计算] 基频...", end=" ", flush=True)
        try:
            # 使用YIN算法提取基频
            self.f0 = librosa.yin(
                self.y, 
                fmin=50, 
                fmax=500, 
                sr=self.sr
            )
            # 提取有效基频值
            self.f0_valid = self.f0[~np.isnan(self.f0)]
            if len(self.f0_valid) == 0:
                self.f0_valid = np.array([100.0])
            print(f"✓ ({len(self.f0_valid)} valid frames)")
        except Exception as e:
            print(f"✗ (使用默认值)")
            self.f0 = np.array([100.0])
            self.f0_valid = self.f0
    
    # --- 辅助计算函数 (保持原逻辑) ---
    def _quantile(self, frequencies, magnitudes, q):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        cumsum = np.cumsum(magnitudes)
        cumsum_normalized = cumsum / cumsum[-1]
        idx = np.searchsorted(cumsum_normalized, q)
        if idx >= len(frequencies): idx = len(frequencies) - 1
        return frequencies[idx] / 1000.0

    # --- 特征计算方法 ---
    def _meanfreq(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        return np.sum(frequencies * magnitudes) / mag_sum / 1000.0
    
    def _sd(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        mean = np.sum(frequencies * magnitudes) / mag_sum
        variance = np.sum(((frequencies - mean) ** 2) * magnitudes) / mag_sum
        return np.sqrt(variance) / 1000.0
    
    def _median(self, frequencies, magnitudes):
        return self._quantile(frequencies, magnitudes, 0.5)
    
    def _Q25(self, frequencies, magnitudes):
        return self._quantile(frequencies, magnitudes, 0.25)
    
    def _Q75(self, frequencies, magnitudes):
        return self._quantile(frequencies, magnitudes, 0.75)
    
    def _IQR(self, frequencies, magnitudes):
        return self._Q75(frequencies, magnitudes) - self._Q25(frequencies, magnitudes)
    
    def _skew(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        mean = np.sum(frequencies * magnitudes) / mag_sum
        variance = np.sum(((frequencies - mean) ** 2) * magnitudes) / mag_sum
        std = np.sqrt(variance)
        if std == 0: return 0.0
        third_moment = np.sum(((frequencies - mean) ** 3) * magnitudes) / mag_sum
        return third_moment / (std ** 3)
    
    def _kurt(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        mean = np.sum(frequencies * magnitudes) / mag_sum
        variance = np.sum(((frequencies - mean) ** 2) * magnitudes) / mag_sum
        std = np.sqrt(variance)
        if std == 0: return 0.0
        fourth_moment = np.sum(((frequencies - mean) ** 4) * magnitudes) / mag_sum
        return (fourth_moment / (std ** 4)) - 3.0
    
    def _sp_ent(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        normalized = magnitudes / mag_sum
        normalized = normalized[normalized > 0]
        return -np.sum(normalized * np.log2(normalized))
    
    def _sfm(self, frequencies, magnitudes):
        if np.sum(magnitudes) == 0: return 0.0
        mag_positive = magnitudes[magnitudes > 0]
        if len(mag_positive) == 0: return 0.0
        geometric_mean = np.exp(np.mean(np.log(mag_positive)))
        arithmetic_mean = np.mean(mag_positive)
        return np.log10(geometric_mean / arithmetic_mean + 1e-10)
    
    def _mode(self, frequencies, magnitudes):
        if np.sum(magnitudes) == 0: return 0.0
        n_bins = min(50, len(magnitudes) // 2)
        if n_bins < 2: n_bins = 2
        counts, bin_edges = np.histogram(frequencies, bins=n_bins, weights=magnitudes)
        mode_bin = np.argmax(counts)
        return (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2.0 / 1000.0
    
    def _centroid(self, frequencies, magnitudes):
        mag_sum = np.sum(magnitudes)
        if mag_sum == 0: return 0.0
        return np.sum(frequencies * magnitudes) / mag_sum / 1000.0
    
    # --- 基频相关特征 (基于整个文件提取) ---
    def _meanfun(self):
        return np.mean(self.f0_valid) / 1000.0
    
    def _minfun(self):
        return np.min(self.f0_valid) / 1000.0
    
    def _maxfun(self):
        return np.max(self.f0_valid) / 1000.0
    
    # --- 显性特征 (逐帧后平均) ---
    def _meandom(self, magnitudes):
        if len(magnitudes) == 0: return 0.0
        threshold = np.percentile(magnitudes, 80)
        dominant_mask = magnitudes >= threshold
        dominant_freqs = self.frequencies[dominant_mask]
        if len(dominant_freqs) == 0: return 0.0
        return np.mean(dominant_freqs) / 1000.0
    
    def _maxdom(self, magnitudes):
        if len(magnitudes) == 0: return 0.0
        threshold = np.percentile(magnitudes, 80)
        dominant_mask = magnitudes >= threshold
        dominant_freqs = self.frequencies[dominant_mask]
        if len(dominant_freqs) == 0: return 0.0
        return np.max(dominant_freqs) / 1000.0

    def _mindom(self, magnitudes):
        if len(magnitudes) == 0: return 0.0
        threshold = np.percentile(magnitudes, 80)
        dominant_mask = magnitudes >= threshold
        dominant_freqs = self.frequencies[dominant_mask]
        if len(dominant_freqs) == 0: return 0.0
        return np.min(dominant_freqs) / 1000.0

    def _dfrange(self, magnitudes):
        return self._maxdom(magnitudes) - self._mindom(magnitudes)
    
    def _modindx(self):
        if len(self.f0_valid) < 2: return 0.0
        diff = np.abs(np.diff(self.f0_valid))
        total_diff = np.sum(diff)
        freq_range = np.max(self.frequencies) - np.min(self.frequencies)
        if freq_range == 0: return 0.0
        return total_diff / freq_range
    
    # ======================== 提取主函数 ========================
    
    def extract_all_features(self):
        """
        提取整个音频文件的平均特征
        返回: pd.DataFrame (单行)
        """
        print(f"[提取] 逐帧特征...", end=" ", flush=True)
        
        frame_features = []
        n_frames = self.magnitude.shape[1]
        
        # 1. 逐帧提取需要统计的特征
        for frame_idx in range(n_frames):
            mag = self.magnitude[:, frame_idx]
            
            # 这里计算每一帧的特征
            frame_dict = {
                'meanfreq': self._meanfreq(self.frequencies, mag),
                'sd': self._sd(self.frequencies, mag),
                'median': self._median(self.frequencies, mag),
                'Q25': self._Q25(self.frequencies, mag),
                'Q75': self._Q75(self.frequencies, mag),
                'IQR': self._IQR(self.frequencies, mag),
                'skew': self._skew(self.frequencies, mag),
                'kurt': self._kurt(self.frequencies, mag),
                'sp.ent': self._sp_ent(self.frequencies, mag),
                'sfm': self._sfm(self.frequencies, mag),
                'mode': self._mode(self.frequencies, mag),
                'centroid': self._centroid(self.frequencies, mag),
                'meandom': self._meandom(mag),
                'maxdom': self._maxdom(mag),
                'mindom': self._mindom(mag),
                'dfrange': self._dfrange(mag)
            }
            frame_features.append(frame_dict)
        
        # 2. 将帧级特征转换为DataFrame进行平均
        df_frames = pd.DataFrame(frame_features)
        
        # 3. 计算整个文件的平均值
        print(f"\r[提取] 聚合文件特征...", end=" ", flush=True)
        global_features = df_frames.mean().to_dict()
        
        # 4. 添加基频和调制指数(这些本来就是基于整个文件或序列的)
        global_features['meanfun'] = self._meanfun()
        global_features['minfun'] = self._minfun()
        global_features['maxfun'] = self._maxfun()
        global_features['modindx'] = self._modindx()
        
        print(f"\r[提取] 完成 ✓ ({n_frames} frames processed)")
        
        # 5. 返回单行 DataFrame
        return pd.DataFrame([global_features])


def main():
    parser = argparse.ArgumentParser(description='音频特征提取与分类')
    parser.add_argument('audio_file', help='输入音频文件 (mp3/wav)')
    parser.add_argument('--model', default='voice_model.json', help='模型路径 (默认: voice_model.json)')
    args = parser.parse_args()

    # 确保文件存在
    if not os.path.exists(args.audio_file):
        print(f"✗ 错误: 文件不存在 - {args.audio_file}")
        sys.exit(1)
        
    if not os.path.exists(args.model) or not os.path.exists('scaler.pkl'):
        print(f"✗ 错误: 模型文件 {args.model} 或 scaler.pkl 不存在")
        sys.exit(1)
    
    print("=" * 70)
    print("分析系统")
    print("=" * 70)
    
    try:
        # 1. 特征提取
        extractor = AudioFeatureExtractor(args.audio_file)
        df_features = extractor.extract_all_features()
        
        # 保存特征以便参考
        feature_file = Path(args.audio_file).stem + "_features.csv"
        df_features.to_csv(feature_file, index=False)
        print(f"[保存] 提取特征到 {feature_file}")
        
        # 2. 加载模型和 Scaler
        print(f"[加载] 加载 scaler.pkl...", end=" ", flush=True)
        scaler = joblib.load('scaler.pkl')
        print("✓")
        
        print(f"[加载] 加载模型 {args.model}...", end=" ", flush=True)
        model = xgb.XGBClassifier()
        model.load_model(args.model)
        print("✓")
        
        # 3. 预测
        print("[预测] 进行特征标准化与分析...", end=" ", flush=True)
        
        # 定义特征顺序 (与训练时保持一致)
        feature_order = [
            'meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm',
            'mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom',
            'dfrange','modindx'
        ]
        
        # --- 权重分组 (依据Gain值) ---
        # 核心层: meanfun (King)
        # 关键层: mode, IQR, Q25, minfun, sfm (拉锯区)
        # 辅助层: 其余14个特征 (基础)
        
        X = df_features[feature_order]
        X_scaled = scaler.transform(X)
        
        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)
        print("✓\n")
        
        # 4. 输出结果
        print("=" * 70)
        print("结果")
        print("=" * 70)
        
        # XGBoost 的 proba 输出通常是 [prob_class0, prob_class1]
        # 需根据模型正类标签决定谁是男谁是女
        male_prob = proba[0][1]
        female_prob = proba[0][0]
        
        print(f"  模型预测: {'女性' if pred[0] == 1 else '男性'}")
        print(f"  预测概率: 男 {male_prob:.2%} / 女 {female_prob:.2%}")

        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()