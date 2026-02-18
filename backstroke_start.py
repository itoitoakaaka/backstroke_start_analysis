import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# バイオメカニクスパラメータの特徴量リスト
features = [
    'Hands-off phase relative time (%)', 'Take-off phase relative time (%)',
    'Flight phase relative time (%)', 'Entry phase relative time (%)',
    'Resultant take-off velocity (m·s-1)', 'Resultant flight velocity (m·s-1)',
    'Resultant entry velocity (m·s-1)', 'Wrist entry angle (o)',
    'Shoulder entry angle (o)', 'Hip entry angle (o)', 'Back arc angle (o)',
    'Upper limb force at starting position (N/N)', 'Maximal upper limb force and time (N/N; %)',
    'Upper limb horizontal and vertical impulse', 'Lower limbs force at starting position (N/N)',
    '1st maximal lower limb force and time (N/N; %)', 'Intermediate lower limb force and time (N/N; %)',
    '2nd maximal lower limb force and time (N/N; %)', 'Lower limb horizontal、 vertical and medio-lateral impulse'
]


def main():
    import sys
    # コマンドライン引数またはデフォルトのCSVパスを使用
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'your_data.csv'
    df = pd.read_csv(csv_path)

    X = df[features].astype(float)
    y = df['5 m start time (s)'].astype(float)

    # --- 線形回帰 ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print(f"Linear Regression Mean Squared Error: {mse_lr}")

    # --- ニューラルネットワーク (ANN) ---
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from sklearn.preprocessing import StandardScaler

        # 特徴量の標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ANNアーキテクチャ定義
        model_ann = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # 出力層（回帰: ニューロン1つ）
        ])

        # モデルのコンパイル
        model_ann.compile(optimizer='adam', loss='mean_squared_error')

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 学習
        model_ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # テストデータで評価
        mse_ann = model_ann.evaluate(X_test, y_test)
        print(f"ANN Mean Squared Error: {mse_ann}")

    except ImportError:
        print("⚠️ TensorFlowがインストールされていません。ANN部分をスキップします。")


if __name__ == "__main__":
    main()
