import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置绘图
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class WaterSampleClustrer:
    def __init__(self, input_dir="data"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_dir = os.path.join(self.base_path, input_dir)
        self.col_map = {'时间戳': 'timestamp', '流量值': 'flow'}

    def extract_features(self, file_path):
        """为每个文件提取画像特征"""
        try:
            # 确保读取逻辑稳健
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_csv(file_path)

            df.columns = [str(c).strip() for c in df.columns]
            df.rename(columns=self.col_map, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp')
            
            # 基础预处理
            flow = pd.to_numeric(df['flow'], errors='coerce').interpolate().ffill().bfill()
            df['date'] = df['timestamp'].dt.date
            
            # --- 特征 1: 逐日基线分析 ---
            inter_base = df.groupby('date')['flow'].median()
            drift_score = inter_base.diff().abs().mean()
            pulse_score = (inter_base - inter_base.median()).abs().max()
            
            # --- 特征 2: 失效分析 ---
            stuck_count = ((df['flow'] == 800) | (df['flow'].diff() == 0)).sum()
            stuck_ratio = stuck_count / len(df)
            
            # --- 特征 3: 频域周期强度 (FFT) ---
            y = flow.values - np.mean(flow.values)
            n = len(y)
            yf = fft(y)
            idx_1cpd = int(n * 300 / 86400)
            periodic_strength = np.abs(yf[idx_1cpd]) if idx_1cpd < n//2 else 0

            return {
                "drift_score": drift_score,
                "pulse_score": pulse_score,
                "stuck_ratio": stuck_ratio,
                "periodic_strength": periodic_strength
            }
        except Exception as e:
            print(f"提取特征失败 {os.path.basename(file_path)}: {e}")
            return None

    def run_clustering(self):
        files = sorted(glob.glob(os.path.join(self.input_dir, "样本*.xlsx")))
        if not files:
            files = sorted(glob.glob(os.path.join(self.input_dir, "样本*.csv")))
            
        file_names = []
        features_list = []

        print(f"正在对 {len(files)} 个样本进行特征提取...")
        for f in files:
            feat = self.extract_features(f)
            if feat:
                file_names.append(os.path.basename(f))
                features_list.append(feat)

        # 1. 构建特征矩阵
        X_df = pd.DataFrame(features_list).fillna(0)

        # 2. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        # 3. K-Means 聚类
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # 4. 可视化：使用 PCA 降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        # 增大散点尺寸 s=300，方便在圆圈内写数字
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                            s=400, edgecolors='k', alpha=0.6)
        
        # ---------------------------------------------------------
        # 修改部分：标注 1-50 的数字
        # ---------------------------------------------------------
        for i, filename in enumerate(file_names):
            # 使用正则表达式提取数字，例如 "样本008.xlsx" -> 8
            match = re.search(r'样本(\d+)', filename)
            if match:
                sample_id = str(int(match.group(1))) # 转为 int 再转 str 消除前导零
            else:
                sample_id = "?" # 若没匹配到数字则显示问号

            plt.annotate(sample_id, 
                         (X_pca[i, 0], X_pca[i, 1]), 
                         fontsize=10, 
                         fontweight='bold',
                         ha='center', 
                         va='center',
                         color='black')
        # ---------------------------------------------------------

        plt.colorbar(scatter, label='类别标签')
        plt.title("50个样本无监督分类分布图 (数字对应样本001-050)", fontsize=14)
        plt.xlabel("主成分 1 (波动特征)", fontsize=12)
        plt.ylabel("主成分 2 (异常特征)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存结果
        plt.savefig("样本分类数字分布图.png", dpi=150)
        
        # 保存 Excel
        result_df = pd.DataFrame({"文件名": file_names, "分类结果": clusters})
        result_df = pd.concat([result_df, X_df], axis=1)
        result_df.to_excel("分类结果汇总.xlsx", index=False)
        
        print("\n任务完成：")
        print("1. [样本分类数字分布图.png] 已生成，图中数字即为样本编号。")
        print("2. [分类结果汇总.xlsx] 已生成。")

if __name__ == "__main__":
    clusterer = WaterSampleClustrer()
    clusterer.run_clustering()