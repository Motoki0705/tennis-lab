# Test Design & Results

## Branch
<branch_name>

---

# 1. Module Responsibilities
<!-- テスト対象モジュールの責務・想定機能を明確化 -->

- Module:
- Responsibilities:
  - A
  - B
  - C

---

# 2. Test Design (before implementation)
<!-- 各チェックボックスに対してどのようなテストを作成するかを先に書く -->
## Plan
### Step 1: <summary>
- Expected behavior:
- Test cases:
  - Case 1:
  - Case 2:
- Level: unit / integration
- Fixtures: existing / new

### Step 2: <summary>
- ...

### Step 3: <summary>
- ...

---

# 3. Test Implementation Log
<!-- 実装フェーズで1ステップごとに追記 -->

## Step 1: <summary>
### Added Tests
<!-- 追加した test_xxx.py、改修、mock/fixture の使用など -->

### Execution
```
uv run pytest tests/unit/<file>.py -q
```

### Result
- PASS / FAIL  
- Notes:

---

## Step 2: <summary>
### Added Tests
### Execution
### Result

---

## Step 3: <summary>
### Added Tests
### Execution
### Result

