# Best Practices

## General Principles

* **Format:** Use **Google style (Napoleon)**. Structure → One-line summary → blank line → Details → Section order.
* **Length:** One-line summary ≤ 80 chars. Detail section **within 3–6 lines**. Include only essential `Args/Returns/Raises/Examples`.
* **Tone:** Present tense, active voice, factual. Do **not** describe implementation (that belongs in `docs/`).
* **Consistency:** Section headers must be in English (`Args`, `Returns`, `Raises`, `Examples`, `Notes`, `Attributes`).
* **Change flow:** “Update `docs/` first, then docstrings.” Keep source of truth centralized.

---

## Module Docstring (top of each file)

Focus only on **purpose** and **public interface**. Avoid internal explanations.

* Structure: `Summary` → one short paragraph (what/why) → `Example` (1 block max) → `Notes` (optional)
* Avoid: long background, history, internal logic
* **Minimal Template:**

  ```python
  """<One-line summary>.

  <Short paragraph describing the module's purpose and I/O boundaries (3–6 lines)>.

  Example:
      >>> <Shortest usage example, 1–3 lines>
  """
  ```

---

## Class Docstring

Describe only **responsibility**, **main arguments**, and **key attributes**.
Document `__init__` parameters under `Args:`.

* Structure: `Summary` → short detail (2–4 lines) → `Args` → `Attributes` (if needed) → `Examples` (optional)
* **Minimal Template:**

  ```python
  class Foo:
      """<One-line summary>.

      <What problem this class solves and its assumptions (2–4 lines)>.

      Args:
          x (int): <Meaning>.
          y (str, optional): <Meaning>. Defaults to "…".

      Attributes:
          size (int): <Meaning of public attribute>.
      """
  ```

---

## Function / Method Docstring

Focus on **inputs, outputs, and exceptions**.
Avoid algorithmic detail.

* Structure: `Summary` → short detail (1–3 lines) → `Args` → `Returns` → `Raises` (only actual ones) → `Examples` (optional)
* **Minimal Template:**

  ```python
  def foo(x: int, y: str = "…") -> bool:
      """<One-line summary>.

      <Preconditions or side effects, if any (1–3 lines)>.

      Args:
          x (int): <Meaning>.
          y (str, optional): <Meaning>. Defaults to "…".

      Returns:
          bool: <Meaning of the boolean value>.

      Raises:
          ValueError: <Condition>.
      """
  ```

---

## Division of Responsibility with `docs/`

* **Docstring:** Local context — purpose, I/O boundaries, exceptions, minimal usage example.
* **docs/**: Global context — specifications, design rationale, constraints, flows, API list, decision records.
* In the module’s `Notes:` section, only reference the spec file path, not the explanation.

  ```
  Notes:
      See docs/specs/<feature>.md
  ```
