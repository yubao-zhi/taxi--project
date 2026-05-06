import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== M1 数据处理模块 =====================
def load_data():
    print("正在加载数据...")
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    df = pd.read_parquet(url)
    df = df.sample(frac=0.05, random_state=42)
    print(f"数据加载完成，共 {len(df)} 行")
    return df

def data_quality_report(df):
    print("\n===== M1 数据质量报告 =====")
    print("缺失值统计：")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n异常值统计（IQR法）：")
    for col in ["trip_distance", "fare_amount"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        out = df[(df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)]
        print(f"{col} 异常值数量：{len(out)}")

def clean_data(df):
    df = df.copy()
    # 1. 删除重复行（避免重复统计）
    df = df.drop_duplicates()
    # 2. 删除关键列缺失（缺失无法分析）
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "fare_amount", "trip_distance"])
    # 3. 过滤异常值（符合真实业务）
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
    return df

def feature_engineering(df):
    df = df.copy()
    df["pickup_time"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["hour"] = df["pickup_time"].dt.hour
    df["weekday"] = df["pickup_time"].dt.dayofweek
    df["is_peak"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)
    # 衍生特征1：行程时长（分钟）
    df["duration_min"] = (pd.to_datetime(df["tpep_dropoff_datetime"]) - df["pickup_time"]).dt.total_seconds() / 60
    # 衍生特征2：单位距离费用
    df["fare_per_km"] = df["fare_amount"] / (df["trip_distance"] + 1e-6)
    print("M1 特征提取完成")
    return df

# ===================== M2 分析可视化模块 =====================
def m2_visualize(df):
    print("\n===== M2 开始绘图并保存到 outputs/ =====")
    import os
    os.makedirs("outputs", exist_ok=True)

    # 1. 小时需求规律
    plt.figure(figsize=(12,5))
    df.groupby("hour").size().plot(kind="line", marker="o")
    plt.title("各小时出行需求量")
    plt.xlabel("小时")
    plt.ylabel("订单数")
    plt.savefig("outputs/1_hour_demand.png", dpi=300)
    plt.close()

    # 2. 区域热度TOP10
    top_zone = df.groupby("PULocationID").size().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12,5))
    top_zone.plot(kind="bar")
    plt.title("上车热点区域TOP10")
    plt.savefig("outputs/2_top_zone.png", dpi=300)
    plt.close()

    # 3. 车费与距离关系
    plt.figure(figsize=(10,5))
    plt.scatter(df["trip_distance"], df["fare_amount"], alpha=0.1)
    plt.xlabel("行程距离")
    plt.ylabel("车费")
    plt.title("距离-车费关系")
    plt.savefig("outputs/3_fare_distance.png", dpi=300)
    plt.close()

    # 4. 自选：高峰vs非高峰车费
    plt.figure(figsize=(8,4))
    df.groupby("is_peak")["fare_amount"].mean().plot(kind="bar")
    plt.title("高峰vs非高峰平均车费")
    plt.savefig("outputs/4_peak_fare.png", dpi=300)
    plt.close()
    print("M2 图表保存完成")

# ===================== M3 预测模型模块 =====================
def m3_model(df):
    print("\n===== M3 需求预测模型 =====")
    # 构造时段-区域需求
    demand = df.groupby(["hour", "PULocationID"]).size().reset_index(name="demand")
    X = demand[["hour", "PULocationID"]]
    y = demand["demand"]

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    import tensorflow as tf
    tf.random.set_seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 随机森林
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    print("随机森林 MAE:", round(mean_absolute_error(y_test, pred_rf),2))
    print("随机森林 RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_rf)),2))

    # 神经网络
    model = tf.keras.Sequential([tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=0)
    pred_nn = model.predict(X_test, verbose=0)
    print("神经网络 MAE:", round(mean_absolute_error(y_test, pred_nn),2))
    print("神经网络 RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_nn)),2))

    plt.figure(figsize=(10,4))
    plt.plot(hist.history["loss"], label="训练loss")
    plt.plot(hist.history["val_loss"], label="验证loss")
    plt.legend()
    plt.title("Loss曲线")
    plt.savefig("outputs/5_loss_curve.png", dpi=300)
    plt.close()

# ===================== M4 智能问答接口 =====================
def m4_qa():
    print("\n===== M4 智能问答系统 =====")
    while True:
        q = input("\n请输入问题（输入 exit 退出）：")
        if q == "exit": break
        if "高峰" in q or "小时" in q:
            print("→ 已为你生成时段需求图：outputs/1_hour_demand.png")
        elif "区域" in q or "热点" in q:
            print("→ 已为你生成区域TOP10图：outputs/2_top_zone.png")
        elif "车费" in q or "钱" in q:
            print("→ 已为你生成车费分析图：outputs/3_fare_distance.png")
        elif "预测" in q or "多少单" in q:
            print("→ 已完成需求预测，查看MAE/RMSE结果")
        elif "模型" in q:
            print("→ 已生成loss曲线：outputs/5_loss_curve.png")
        else:
            print("→ 支持查询：高峰时段、热点区域、车费、需求预测、模型对比")

# ===================== 主函数：一键运行 =====================
def main():
    df = load_data()
    data_quality_report(df)
    df_clean = clean_data(df)
    df_feat = feature_engineering(df_clean)
    m2_visualize(df_feat)
    m3_model(df_feat)
    m4_qa()

if __name__ == "__main__":
    main()