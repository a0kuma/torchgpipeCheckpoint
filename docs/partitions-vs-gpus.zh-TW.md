# 分區與 GPU：理解兩者的關係

## 問題：分區（partition）的數量一定要跟 GPU 的數量一樣嗎？

**簡答**：不，分區的數量**不需要**與 GPU 的數量相同。

## 核心概念

### 分區（Partitions）
**分區**是模型中連續層的集合，這些層會一起處理。當你指定 `balance=[2, 2, 3]` 時，你會建立 3 個分區，分別包含 2、2 和 3 層。

### 裝置（Devices/GPUs）
**裝置**是執行計算的實體 GPU（或 CPU）。你可以透過 `devices` 參數明確指定裝置，或讓 torchgpipe 自動使用所有可用的 CUDA 裝置。

## 兩者的關係

關鍵規則是：**分區數量必須小於或等於裝置數量。**

```
分區數量 ≤ 裝置數量
```

### 為什麼有這樣的彈性？

1. **多個分區可以共用同一個裝置**：如果你有 4 個分區但只有 2 個 GPU，torchgpipe 會將多個分區放在同一個 GPU 上。

2. **不是所有裝置都需要使用**：如果你有 8 個 GPU 但只有 3 個分區，只有前 3 個 GPU 會被使用。

## 範例

### 範例 1：分區數量等於 GPU 數量（最常見）

```python
from torchgpipe import GPipe

model = nn.Sequential(a, b, c, d)
# 4 個分區在 4 個 GPU 上（cuda:0, cuda:1, cuda:2, cuda:3）
model = GPipe(model, balance=[1, 1, 1, 1], chunks=8)

# 結果：
# 分區 0（層 a）→ cuda:0
# 分區 1（層 b）→ cuda:1
# 分區 2（層 c）→ cuda:2
# 分區 3（層 d）→ cuda:3
```

### 範例 2：分區少於 GPU

```python
model = nn.Sequential(a, b, c, d)
# 2 個分區在 4 個可用的 GPU 上
model = GPipe(model, balance=[2, 2], chunks=8)

# 結果：
# 分區 0（層 a, b）→ cuda:0
# 分區 1（層 c, d）→ cuda:1
# cuda:2 和 cuda:3 不會被使用
```

### 範例 3：分區多於 GPU（使用特定裝置）

```python
model = nn.Sequential(a, b, c, d)
# 4 個分區在 2 個 GPU 上 - 明確指定裝置
model = GPipe(model, balance=[1, 1, 1, 1], devices=['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1'], chunks=8)

# 結果：
# 分區 0（層 a）→ cuda:0
# 分區 1（層 b）→ cuda:1
# 分區 2（層 c）→ cuda:0
# 分區 3（層 d）→ cuda:1
```

### 範例 4：單一裝置（CPU）配多個分區

```python
model = nn.Sequential(a, b)
# 2 個分區在 1 個 CPU 裝置上
model = GPipe(model, balance=[1, 1], devices=['cpu', 'cpu'], chunks=4)

# 結果：
# 分區 0（層 a）→ cpu
# 分區 1（層 b）→ cpu
```

## 實際基準測試範例

來自本專案中 ResNet-101 速度基準測試：

```python
# Pipeline-8 配置
balance = [26, 22, 33, 44, 44, 66, 66, 69]  # 8 個分區
model = GPipe(model, balance, devices=devices, chunks=chunks)
```

如果 `devices` 包含 8 個 GPU ID，每個分區會有自己的 GPU。如果只包含 4 個 GPU ID，多個分區會共用 GPU。

## 自動裝置分配

當你不指定 `devices` 參數時，torchgpipe 會自動使用可用的 CUDA 裝置：

```python
# 來自 gpipe.py 第 244-246 行
if devices is None:
    devices = range(torch.cuda.device_count())
```

這意味著：
- 如果你有 4 個 GPU 和 4 個分區 → 每個分區獲得一個 GPU
- 如果你有 4 個 GPU 和 2 個分區 → 只使用前 2 個 GPU
- 如果你有 2 個 GPU 和 4 個分區 → **錯誤！**（裝置太少）

## 錯誤處理

如果你嘗試建立比裝置數量更多的分區，而沒有明確指定裝置重用，你會得到錯誤：

```python
# 來自 gpipe.py 第 100-102 行
if len(balance) > len(devices):
    raise IndexError('too few devices to hold given partitions '
                     f'(devices: {len(devices)}, partitions: {len(balance)})')
```

**避免此錯誤的方法**：要麼減少分區數量，要麼明確提供一個會重用 GPU 的裝置列表。

## 最佳實踐

1. **最大平行度**：讓分區數量等於你擁有的 GPU 數量。

2. **記憶體限制**：如果想簡化管線，使用比 GPU 數量更少的分區。

3. **在有限硬體上測試**：在 `devices` 參數中明確指定裝置重用。

4. **最佳效能**：使用自動平衡工具來決定分區大小：

```python
from torchgpipe.balance import balance_by_time

partitions = torch.cuda.device_count()  # 匹配 GPU 數量
sample = torch.rand(128, 3, 224, 224)
balance = balance_by_time(partitions, model, sample)

model = GPipe(model, balance, chunks=8)
```

## 總結

- ✅ 分區可以等於 GPU 數量（最常見，最佳平行度）
- ✅ 分區可以少於 GPU 數量（有未使用的 GPU）
- ✅ 分區可以多於 GPU 數量（需明確指定裝置）
- ❌ 分區不能超過裝置數量（除非明確指定裝置重用）

這種彈性讓你可以：
- 在單一 GPU 上測試多分區模型
- 優化記憶體使用和計算分配
- 在不同的硬體配置上擴展模型
