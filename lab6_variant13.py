"""
Лабораторная работа № 6. Вариант 13.
Кугушева Арина, группа БИ2-1. Дисциплина: Теория алгоритмов.

ЛИНЕЙНЫЕ СТРУКТУРЫ ДАННЫХ И ТРУДОЁМКОСТЬ ИХ ОПЕРАЦИЙ

Условия варианта 13:
  * структуры ............... двусвязный список и очередь на кольцевом буфере;
  * исследуемые операции .... вставка в середину, обход в обе стороны;
  * размеры n ............... 1000, 2000, 4000;
  * прикладная задача ....... плей-лист с переходами («следующий» / «предыдущий»);
  * доп. требования ......... Т1 (проверка на малом примере), Т2 (сравнение со
                              встроенным list), Т3 (график T(n) и оценка),
                              Т4 (память на 10 000 элементов, sys.getsizeof).

КАК УСТРОЕН ФАЙЛ (читать сверху вниз):
  Часть 1. Константы.
  Часть 2. Node и DoublyLinkedList — двусвязный список.
  Часть 3. CircularQueue — очередь на кольцевом буфере.
  Часть 4. Playlist — прикладная задача (плей-лист на двусвязном списке).
  Часть 5. Измерительные функции (timed и др.).
  Часть 6. Эксперименты (по одному на каждую исследуемую операцию).
  Часть 7. Проверка корректности (Т1) и самотесты.
  Часть 8. Вывод результатов, графики, main().

ЗАПУСК:
    python lab6_variant13.py                      # демонстрация + все измерения
    python lab6_variant13.py --plots plots        # + графики (нужен matplotlib)
    python lab6_variant13.py --json results.json  # + результаты в JSON
"""
import argparse
import json
import random
import statistics
import sys
import time
import tracemalloc
from collections import deque

# ======================================================================
# ЧАСТЬ 1. КОНСТАНТЫ
# ======================================================================

SIZES = [1000, 2000, 4000]   # размеры n из условия варианта
MIDDLE_OPS = 200             # сколько вставок в середину делаем в каждом измерении
REPEATS = 7                  # сколько раз повторяем измерение (берём МЕДИАНУ)
TRANSITIONS = 10_000         # сколько переходов «следующий/предыдущий» в плей-листе
MEMORY_N = 10_000            # размер для оценки памяти (требование Т4)
SEED = 42                    # зерно датчика случайных чисел -> результаты воспроизводимы


# ======================================================================
# ЧАСТЬ 2. ДВУСВЯЗНЫЙ СПИСОК
# ======================================================================

class Node:
    """Узел двусвязного списка: значение и ДВЕ ссылки (вперёд и назад).

        None <- [prev | value | next] <-> [prev | value | next] -> None

    __slots__ говорит Python хранить в объекте только эти три поля
    (без словаря атрибутов): узлов много, экономим память.
    """
    __slots__ = ("value", "prev", "next")

    def __init__(self, value, prev=None, nxt=None):
        self.value = value
        self.prev = prev     # ссылка на предыдущий узел (у головы — None)
        self.next = nxt      # ссылка на следующий узел (у хвоста — None)


class DoublyLinkedList:
    """Двусвязный список с ссылками на голову (head) и хвост (tail).

    Главная идея: если у нас УЖЕ ЕСТЬ ссылка на узел, то вставить рядом или
    удалить его можно за Θ(1) — достаточно переставить 2-4 ссылки.
    А вот ДОБРАТЬСЯ до узла с номером i приходится обходом — это Θ(n).
    """

    def __init__(self):
        self.head = None   # первый узел (None, если список пуст)
        self.tail = None   # последний узел
        self.size = 0      # число элементов (чтобы len() работал за Θ(1))

    # ---------- вставка рядом с известным узлом — всегда Θ(1) ----------

    def insert_after(self, node, value):
        """Вставить value сразу ПОСЛЕ узла node. Θ(1).

        Было:   node <-> right
        Станет: node <-> new <-> right
        Порядок присваиваний важен: сначала настраиваем новый узел,
        потом переключаем ссылки соседей — иначе потеряем right.
        """
        right = node.next
        new = Node(value, prev=node, nxt=right)   # у нового узла сразу обе ссылки
        node.next = new                            # node теперь смотрит на new
        if right is None:
            self.tail = new                        # вставили после хвоста -> новый хвост
        else:
            right.prev = new                       # right смотрит назад на new
        self.size += 1
        return new                                 # возвращаем узел — им можно пользоваться дальше

    def insert_before(self, node, value):
        """Вставить value сразу ПЕРЕД узлом node. Θ(1). Зеркально к insert_after."""
        left = node.prev
        new = Node(value, prev=left, nxt=node)
        node.prev = new
        if left is None:
            self.head = new                        # вставили перед головой -> новая голова
        else:
            left.next = new
        self.size += 1
        return new

    def push_front(self, value):
        """Вставка в начало. Θ(1)."""
        if self.head is None:                      # список пуст
            self.head = self.tail = Node(value)
            self.size = 1
            return self.head
        return self.insert_before(self.head, value)

    def push_back(self, value):
        """Вставка в конец. Θ(1) — благодаря ссылке tail."""
        if self.tail is None:
            return self.push_front(value)
        return self.insert_after(self.tail, value)

    # ---------- доступ по номеру — Θ(n) ----------

    def node_at(self, index):
        """Найти узел с номером index (нумерация с нуля). Θ(n).

        Оптимизация, которую даёт вторая ссылка: если индекс в первой
        половине, идём от головы, иначе — от хвоста. Это сокращает путь
        вдвое, но порядок остаётся Θ(n): для середины всё равно n/2 шагов.
        """
        if not 0 <= index < self.size:
            raise IndexError("Индекс вне диапазона")
        if index <= self.size // 2:
            node = self.head
            for _ in range(index):                 # index шагов вперёд
                node = node.next
        else:
            node = self.tail
            for _ in range(self.size - 1 - index): # (size-1-index) шагов назад
                node = node.prev
        return node

    def get(self, index):
        """Значение элемента с номером index. Θ(n)."""
        return self.node_at(index).value

    def insert(self, index, value):
        """ВСТАВКА ПО НОМЕРУ (в том числе в середину) — исследуемая операция.

        Стоимость = поиск позиции Θ(n) + перестановка ссылок Θ(1) = Θ(n).
        После вставки новый элемент имеет номер index.
        """
        if not 0 <= index <= self.size:
            raise IndexError("Индекс вне диапазона")
        if index == self.size:                     # вставка в конец
            return self.push_back(value)
        # Находим узел, который сейчас стоит на месте index, и вставляем ПЕРЕД ним.
        return self.insert_before(self.node_at(index), value)

    # ---------- удаление ----------

    def remove_node(self, node):
        """Удалить узел node (ссылка на него уже есть). Θ(1).

        Соседи «перешагивают» через node: left.next = right, right.prev = left.
        """
        left, right = node.prev, node.next
        if left is None:
            self.head = right                      # удалили голову
        else:
            left.next = right
        if right is None:
            self.tail = left                       # удалили хвост
        else:
            right.prev = left
        node.prev = node.next = None               # отцепляем удалённый узел
        self.size -= 1
        return node.value

    def pop_front(self):
        """Удалить и вернуть первый элемент. Θ(1)."""
        if self.head is None:
            raise IndexError("Список пуст")
        return self.remove_node(self.head)

    def pop_back(self):
        """Удалить и вернуть последний элемент. Θ(1) (у односвязного было бы Θ(n))."""
        if self.tail is None:
            raise IndexError("Список пуст")
        return self.remove_node(self.tail)

    # ---------- ОБХОД В ОБЕ СТОРОНЫ — исследуемая операция ----------

    def walk_forward(self):
        """Обход от головы к хвосту по ссылкам next. Θ(n). Возвращает число посещённых узлов."""
        count, node = 0, self.head
        while node is not None:
            count += 1
            node = node.next
        return count

    def walk_backward(self):
        """Обход от хвоста к голове по ссылкам prev. Θ(n).
        Именно ЭТО односвязный список не умеет без разворота."""
        count, node = 0, self.tail
        while node is not None:
            count += 1
            node = node.prev
        return count

    def to_list(self):
        """Содержимое слева направо (для вывода и проверок)."""
        result, node = [], self.head
        while node is not None:
            result.append(node.value)
            node = node.next
        return result

    def to_list_reversed(self):
        """Содержимое справа налево — обход по prev."""
        result, node = [], self.tail
        while node is not None:
            result.append(node.value)
            node = node.prev
        return result

    def find(self, value):
        """Линейный поиск значения. Θ(n). Возвращает узел или None."""
        node = self.head
        while node is not None:
            if node.value == value:
                return node
            node = node.next
        return None

    def check_links(self):
        """Проверка целостности: прямой и обратный обходы дают один и тот же
        набор в противоположном порядке, у головы нет prev, у хвоста нет next."""
        forward, backward = self.to_list(), self.to_list_reversed()
        ok = forward == backward[::-1] and len(forward) == self.size
        if self.head is not None:
            ok = ok and self.head.prev is None and self.tail.next is None
        else:
            ok = ok and self.tail is None
        return ok

    def __len__(self):
        return self.size


# ======================================================================
# ЧАСТЬ 3. ОЧЕРЕДЬ НА КОЛЬЦЕВОМ БУФЕРЕ
# ======================================================================

class CircularQueue:
    """Очередь FIFO на массиве фиксированной ёмкости, замкнутом «в кольцо».

    Храним: массив data, индекс головы head (откуда извлекаем) и количество
    элементов count. Индекс хвоста НЕ хранится, а вычисляется:
        tail = (head + count) mod capacity   — первая свободная ячейка.
    Операция «mod» и есть «замыкание в кольцо»: после последней ячейки
    идёт нулевая. Элементы не сдвигаются -> обе операции Θ(1).
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Ёмкость должна быть положительной")
        self.data = [None] * capacity
        self.capacity = capacity
        self.head = 0       # индекс элемента, который будет извлечён следующим
        self.count = 0      # сколько элементов сейчас в очереди

    def enqueue(self, value):
        """Поставить в очередь. Θ(1)."""
        if self.count == self.capacity:
            raise OverflowError("Очередь переполнена")
        tail = (self.head + self.count) % self.capacity   # куда писать
        self.data[tail] = value
        self.count += 1

    def dequeue(self):
        """Извлечь из очереди. Θ(1)."""
        if self.count == 0:
            raise IndexError("Очередь пуста")
        value = self.data[self.head]
        self.data[self.head] = None                        # чтобы не держать ссылку
        self.head = (self.head + 1) % self.capacity        # голова шагает по кольцу
        self.count -= 1
        return value

    def peek(self):
        """Посмотреть первый элемент, не извлекая. Θ(1)."""
        if self.count == 0:
            raise IndexError("Очередь пуста")
        return self.data[self.head]

    def to_list(self):
        """Содержимое от головы к хвосту (с учётом перехода через конец массива)."""
        return [self.data[(self.head + i) % self.capacity] for i in range(self.count)]

    def __len__(self):
        return self.count


# ======================================================================
# ЧАСТЬ 4. ПРИКЛАДНАЯ ЗАДАЧА — ПЛЕЙ-ЛИСТ С ПЕРЕХОДАМИ
# ======================================================================

class Playlist:
    """Плей-лист: двусвязный список треков + указатель на текущий трек
    + очередь «Играть следующим» на кольцевом буфере.

    Почему именно такие структуры:
      * «следующий» / «предыдущий» трек — один шаг по ссылке next/prev: Θ(1);
      * в конце списка переход «вперёд» возвращает к первому треку (по кругу);
      * добавить трек сразу после текущего — insert_after: Θ(1);
      * удалить текущий трек — remove_node: Θ(1);
      * вставка трека на позицию по номеру — Θ(n) (надо дойти до позиции);
      * очередь «играть следующим» — FIFO с ограниченной ёмкостью.
    """

    def __init__(self, queue_capacity=5):
        self.tracks = DoublyLinkedList()
        self.current = None                          # узел с текущим треком
        self.up_next = CircularQueue(queue_capacity)

    def add(self, title):
        """Добавить трек в конец плей-листа."""
        node = self.tracks.push_back(title)
        if self.current is None:                     # первый трек сразу становится текущим
            self.current = node

    def insert_at(self, position, title):
        """Вставить трек на позицию position (по номеру). Θ(n)."""
        node = self.tracks.insert(position, title)
        if self.current is None:
            self.current = node

    def add_after_current(self, title):
        """Вставить трек сразу после текущего. Θ(1): узел уже известен."""
        if self.current is None:
            self.add(title)
        else:
            self.tracks.insert_after(self.current, title)

    def next_track(self):
        """Следующий трек. Θ(1). После последнего — первый (по кругу)."""
        nxt = self.current.next
        self.current = nxt if nxt is not None else self.tracks.head
        return self.current.value

    def prev_track(self):
        """Предыдущий трек. Θ(1). Перед первым — последний (по кругу)."""
        prv = self.current.prev
        self.current = prv if prv is not None else self.tracks.tail
        return self.current.value

    def remove_current(self):
        """Удалить текущий трек; текущим становится следующий (по кругу). Θ(1)."""
        node = self.current
        nxt = node.next if node.next is not None else self.tracks.head
        self.tracks.remove_node(node)
        self.current = None if len(self.tracks) == 0 else (nxt if nxt is not node else None)
        return node.value

    def queue_up(self, title):
        """Поставить трек в очередь «Играть следующим»."""
        self.up_next.enqueue(title)

    def play_from_queue(self):
        """Взять трек из очереди «Играть следующим»: он вставляется после текущего и
        становится текущим. Если очередь пуста — обычный переход вперёд."""
        if len(self.up_next) == 0:
            return self.next_track()
        title = self.up_next.dequeue()
        self.current = self.tracks.insert_after(self.current, title)
        return title


# ======================================================================
# ЧАСТЬ 5. ИЗМЕРИТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def timed(setup, action, repeats=REPEATS):
    """Время action(объект), с — МЕДИАНА из repeats запусков.

    setup() создаёт СВЕЖИЙ объект перед каждым запуском (его построение
    в измеряемое время не входит). Медиана устойчива к случайным скачкам
    времени (например, когда процессор отвлёкся на другую программу).
    """
    times = []
    for _ in range(repeats):
        obj = setup()
        start = time.perf_counter()
        action(obj)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def build_dll(n):
    """Двусвязный список из чисел 0..n-1."""
    lst = DoublyLinkedList()
    for i in range(n):
        lst.push_back(i)
    return lst


# ======================================================================
# ЧАСТЬ 6. ЭКСПЕРИМЕНТЫ
# ======================================================================

def experiment_middle_insert():
    """Вставка в середину: MIDDLE_OPS вставок в структуру из n элементов.

    Три способа:
      * DLL по номеру    — insert(n//2, x): поиск позиции Θ(n) + Θ(1);
      * DLL по узлу      — insert_after(mid_node, x): узел найден заранее -> Θ(1);
      * встроенный list  — list.insert(n//2, x): сдвиг хвоста массива Θ(n),
                           но сдвиг делает быстрый C-код (memmove).
    """
    rows = []
    for n in SIZES:
        mid = n // 2

        def ops_dll_index(lst):
            for i in range(MIDDLE_OPS):
                lst.insert(mid, i)

        def ops_dll_node(lst_and_node):
            lst, node = lst_and_node
            for i in range(MIDDLE_OPS):
                lst.insert_after(node, i)

        def setup_node():
            lst = build_dll(n)
            return lst, lst.node_at(mid)       # поиск узла НЕ входит в измерение

        def ops_py(arr):
            for i in range(MIDDLE_OPS):
                arr.insert(mid, i)

        rows.append({
            "n": n,
            "dll_index": timed(lambda: build_dll(n), ops_dll_index),
            "dll_node": timed(setup_node, ops_dll_node),
            "list": timed(lambda: list(range(n)), ops_py),
        })
    return rows


def experiment_traversal():
    """Полный обход структуры из n элементов вперёд и назад. Θ(n) для всех."""
    rows = []

    def py_forward(arr):
        for _ in arr:
            pass

    def py_backward(arr):
        for _ in reversed(arr):
            pass

    for n in SIZES:
        rows.append({
            "n": n,
            "dll_forward": timed(lambda: build_dll(n), lambda s: s.walk_forward()),
            "dll_backward": timed(lambda: build_dll(n), lambda s: s.walk_backward()),
            "list_forward": timed(lambda: list(range(n)), py_forward),
            "list_backward": timed(lambda: list(range(n)), py_backward),
        })
    return rows


def experiment_playlist():
    """Переходы в плей-листе: TRANSITIONS шагов «вперёд» и столько же «назад».
    Один переход = одна смена ссылки -> время НЕ зависит от n (Θ(1))."""
    rows = []
    for n in SIZES:
        def setup():
            pl = Playlist()
            for i in range(n):
                pl.add(f"Трек {i}")
            return pl

        def forward(pl):
            for _ in range(TRANSITIONS):
                pl.next_track()

        def backward(pl):
            for _ in range(TRANSITIONS):
                pl.prev_track()

        rows.append({"n": n, "forward": timed(setup, forward), "backward": timed(setup, backward)})
    return rows


def experiment_queue():
    """Дополнительно: очередь на кольцевом буфере против list.pop(0).
    n постановок, затем n извлечений. Кольцо: Θ(n) всего; pop(0): Θ(n²)."""
    rows = []
    for n in SIZES:
        def ring(_):
            q = CircularQueue(n)
            for i in range(n):
                q.enqueue(i)
            for _ in range(n):
                q.dequeue()

        def naive(_):
            arr = []
            for i in range(n):
                arr.append(i)
            for _ in range(n):
                arr.pop(0)

        rows.append({"n": n, "ring": timed(lambda: None, ring), "pop0": timed(lambda: None, naive)})
    return rows


def experiment_memory(n=MEMORY_N):
    """Т4. Расход памяти на n элементов.

    sys.getsizeof(объект) возвращает размер ТОЛЬКО самого объекта (для списка —
    массив указателей, без объектов-значений). Поэтому для двусвязного списка
    считаем: n узлов * размер одного узла. Сами числа-значения у всех структур
    одинаковы, поэтому в сравнение не входят. Для проверки замеряем фактический
    прирост памяти через tracemalloc.
    """
    node_size = sys.getsizeof(Node(0))
    values = list(range(n))                       # общие значения для всех структур

    def traced(builder):
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        obj = builder()
        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        del obj
        return after - before

    def make_dll():
        lst = DoublyLinkedList()
        for v in values:
            lst.push_back(v)
        return lst

    def make_py():
        arr = []
        for v in values:
            arr.append(v)
        return arr

    def make_ring():
        q = CircularQueue(n)
        for v in values:
            q.enqueue(v)
        return q

    arr_exact = list(values)                      # список без запаса ёмкости
    return {
        "n": n,
        "node_bytes": node_size,
        "dll_getsizeof": node_size * n,
        "list_getsizeof": sys.getsizeof(arr_exact),
        "ring_getsizeof": sys.getsizeof(CircularQueue(n).data),
        "dll_traced": traced(make_dll),
        "list_traced": traced(make_py),
        "ring_traced": traced(make_ring),
        "int_bytes": sys.getsizeof(10_000),
    }


# ======================================================================
# ЧАСТЬ 7. ПРОВЕРКА КОРРЕКТНОСТИ (Т1) И САМОТЕСТЫ
# ======================================================================

def demo_small(out=print):
    """Т1. Малый пример: содержимое после каждой операции."""
    out("--- Двусвязный список ---")
    lst = DoublyLinkedList()
    for v in [10, 25, 7]:
        lst.push_back(v)
        out(f"push_back({v:>2}):      {lst.to_list()}")
    lst.push_front(99)
    out(f"push_front(99):     {lst.to_list()}")
    lst.insert(2, 55)
    out(f"insert(2, 55):      {lst.to_list()}   <- вставка в середину")
    lst.insert(len(lst), 1)
    out(f"insert(конец, 1):   {lst.to_list()}")
    out(f"обход вперёд:       {lst.to_list()}")
    out(f"обход назад:        {lst.to_list_reversed()}")
    lst.pop_back()
    out(f"pop_back():         {lst.to_list()}")
    lst.pop_front()
    out(f"pop_front():        {lst.to_list()}")
    out(f"get(2) = {lst.get(2)}, find(7) -> узел со значением {lst.find(7).value}, "
        f"целостность ссылок: {lst.check_links()}")
    out("\n--- Кольцевая очередь (ёмкость 5) ---")
    q = CircularQueue(5)
    for v in ["A", "B", "C", "D", "E"]:
        q.enqueue(v)
    out(f"после 5 enqueue:    head={q.head}, count={q.count}, массив={q.data}")
    out(f"dequeue() -> {q.dequeue()}, dequeue() -> {q.dequeue()}, dequeue() -> {q.dequeue()}")
    q.enqueue("F")
    q.enqueue("G")
    out(f"enqueue F, G (запись пошла в начало массива): head={q.head}, count={q.count}, "
        f"массив={q.data}")
    out(f"порядок извлечения: {q.to_list()}")
    try:
        q.enqueue("X"); q.enqueue("Y")
    except OverflowError as e:
        out(f"при переполнении: {e}")
    out("\n--- Плей-лист ---")
    pl = Playlist(queue_capacity=3)
    for t in ["Альфа", "Бета", "Гамма", "Дельта"]:
        pl.add(t)
    out(f"треки: {pl.tracks.to_list()}, играет: {pl.current.value}")
    out(f"next  -> {pl.next_track()}, next -> {pl.next_track()}, next -> {pl.next_track()}, "
        f"next -> {pl.next_track()}   (после последнего — снова первый)")
    out(f"prev  -> {pl.prev_track()}   (перед первым — последний)")
    pl.add_after_current("Эпсилон")
    out(f"add_after_current('Эпсилон'): {pl.tracks.to_list()}")
    pl.insert_at(2, "Зета")
    out(f"insert_at(2, 'Зета'):         {pl.tracks.to_list()}")
    pl.queue_up("Тэта"); pl.queue_up("Йота")
    out(f"очередь «играть следующим»: {pl.up_next.to_list()}; play_from_queue -> {pl.play_from_queue()}")
    out(f"remove_current() удалил «{pl.remove_current()}», теперь играет «{pl.current.value}»; "
        f"треки: {pl.tracks.to_list()}")


def check_correctness():
    """Самотесты: случайные операции сверяются со встроенными list/deque."""
    rng = random.Random(SEED)
    # --- двусвязный список против list
    lst, ref = DoublyLinkedList(), []
    for step in range(3000):
        op = rng.choice(["front", "back", "insert", "pop_front", "pop_back", "get"])
        if op == "front":
            lst.push_front(step); ref.insert(0, step)
        elif op == "back":
            lst.push_back(step); ref.append(step)
        elif op == "insert":
            i = rng.randint(0, len(ref)); lst.insert(i, step); ref.insert(i, step)
        elif op == "pop_front" and ref:
            assert lst.pop_front() == ref.pop(0)
        elif op == "pop_back" and ref:
            assert lst.pop_back() == ref.pop()
        elif op == "get" and ref:
            i = rng.randrange(len(ref)); assert lst.get(i) == ref[i]
        assert len(lst) == len(ref)
    assert lst.to_list() == ref and lst.to_list_reversed() == ref[::-1] and lst.check_links()
    # --- кольцевая очередь против deque (с многократным переходом через конец массива)
    q, dq = CircularQueue(7), deque()
    for step in range(5000):
        if rng.random() < 0.5 and len(dq) < 7:
            q.enqueue(step); dq.append(step)
        elif dq:
            assert q.dequeue() == dq.popleft()
        assert len(q) == len(dq) and q.to_list() == list(dq)
    # --- плей-лист: круговые переходы
    pl = Playlist()
    for i in range(5):
        pl.add(i)
    seen = [pl.next_track() for _ in range(12)]
    assert seen == [(i + 1) % 5 for i in range(12)]
    seen_back = [pl.prev_track() for _ in range(7)]
    assert pl.tracks.check_links() and len(seen_back) == 7
    # --- вставка в середину совпадает со встроенным list
    for n in (0, 1, 2, 5, 10):
        a, b = DoublyLinkedList(), list(range(n))
        for i in range(n):
            a.push_back(i)
        a.insert(n // 2, "x"); b.insert(n // 2, "x")
        assert a.to_list() == b and a.check_links()
    return True


# ======================================================================
# ЧАСТЬ 8. ВЫВОД, ГРАФИКИ, MAIN
# ======================================================================

def fmt(x, digits=5):
    """Число с десятичной запятой (как принято в русских отчётах)."""
    return f"{x:.{digits}f}".replace(".", ",")


def ratios(rows, key):
    """Отношения T(2n)/T(n) для соседних размеров (у нас n удваивается)."""
    return [rows[i + 1][key] / rows[i][key] for i in range(len(rows) - 1)]


def print_report(res):
    print("\n=== Вставка в середину: %d вставок в структуру из n элементов, с ===" % MIDDLE_OPS)
    print(f"{'n':>6} {'DLL по номеру':>15} {'DLL по узлу':>13} {'list.insert':>13}")
    for r in res["middle"]:
        print(f"{r['n']:>6} {fmt(r['dll_index']):>15} {fmt(r['dll_node']):>13} {fmt(r['list']):>13}")
    print("\n=== Полный обход, с ===")
    print(f"{'n':>6} {'DLL вперёд':>12} {'DLL назад':>11} {'list вперёд':>13} {'list назад':>12}")
    for r in res["traversal"]:
        print(f"{r['n']:>6} {fmt(r['dll_forward']):>12} {fmt(r['dll_backward']):>11} "
              f"{fmt(r['list_forward']):>13} {fmt(r['list_backward']):>12}")
    print("\n=== Плей-лист: %d переходов, с ===" % TRANSITIONS)
    print(f"{'n':>6} {'вперёд':>10} {'назад':>10}")
    for r in res["playlist"]:
        print(f"{r['n']:>6} {fmt(r['forward']):>10} {fmt(r['backward']):>10}")
    print("\n=== Очередь: n постановок + n извлечений, с ===")
    print(f"{'n':>6} {'кольцо':>10} {'list.pop(0)':>13}")
    for r in res["queue"]:
        print(f"{r['n']:>6} {fmt(r['ring']):>10} {fmt(r['pop0']):>13}")
    m = res["memory"]
    print(f"\n=== Память на {m['n']} элементов, байт ===")
    print(f"узел списка: {m['node_bytes']} Б; DLL (узлы): {m['dll_getsizeof']}; "
          f"list: {m['list_getsizeof']}; кольцо: {m['ring_getsizeof']}")
    print(f"tracemalloc: DLL {m['dll_traced']}; list {m['list_traced']}; кольцо {m['ring_traced']}")
    print("\n=== Отношения T(2n)/T(n) ===")
    for name, rows, key in [("DLL вставка по номеру", res["middle"], "dll_index"),
                            ("DLL вставка по узлу", res["middle"], "dll_node"),
                            ("list вставка в середину", res["middle"], "list"),
                            ("DLL обход вперёд", res["traversal"], "dll_forward"),
                            ("DLL обход назад", res["traversal"], "dll_backward"),
                            ("плей-лист: переходы вперёд", res["playlist"], "forward")]:
        print(f"{name:<28}", [fmt(x, 2) for x in ratios(rows, key)])


def save_plots(res, folder):
    """Три графика: вставка в середину, обход, переходы в плей-листе."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    ns = SIZES

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ns, [r["dll_index"] * 1e3 for r in res["middle"]], "o-", label="DLL: insert по номеру")
    ax.plot(ns, [r["list"] * 1e3 for r in res["middle"]], "s-", label="list.insert")
    ax.plot(ns, [r["dll_node"] * 1e3 for r in res["middle"]], "^-", label="DLL: insert_after (узел известен)")
    ax.set_xlabel("n, число элементов"); ax.set_ylabel(f"время {MIDDLE_OPS} вставок, мс")
    ax.set_title("Вставка в середину"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{folder}/middle_insert.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ns, [r["dll_forward"] * 1e3 for r in res["traversal"]], "o-", label="DLL вперёд (next)")
    ax.plot(ns, [r["dll_backward"] * 1e3 for r in res["traversal"]], "o--", label="DLL назад (prev)")
    ax.plot(ns, [r["list_forward"] * 1e3 for r in res["traversal"]], "s-", label="list вперёд")
    ax.plot(ns, [r["list_backward"] * 1e3 for r in res["traversal"]], "s--", label="list назад (reversed)")
    ax.set_xlabel("n, число элементов"); ax.set_ylabel("время полного обхода, мс")
    ax.set_title("Обход в обе стороны"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{folder}/traversal.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ns, [r["forward"] * 1e3 for r in res["playlist"]], "o-", label="переходы «следующий»")
    ax.plot(ns, [r["backward"] * 1e3 for r in res["playlist"]], "s-", label="переходы «предыдущий»")
    ax.set_ylim(0, max(max(r["forward"], r["backward"]) for r in res["playlist"]) * 1e3 * 1.6)
    ax.set_xlabel("n, число треков"); ax.set_ylabel(f"время {TRANSITIONS} переходов, мс")
    ax.set_title("Плей-лист: время переходов не зависит от n"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{folder}/playlist.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ЛР6, вариант 13")
    parser.add_argument("--plots", metavar="DIR", help="папка для графиков")
    parser.add_argument("--json", metavar="FILE", help="файл для результатов в JSON")
    args = parser.parse_args()

    print("=== Т1. Проверка на малом примере ===")
    demo_small()
    check_correctness()
    print("\nСамотесты пройдены (списки и очередь совпадают с list/deque на случайных операциях).")

    res = {
        "middle": experiment_middle_insert(),
        "traversal": experiment_traversal(),
        "playlist": experiment_playlist(),
        "queue": experiment_queue(),
        "memory": experiment_memory(),
        "params": {"sizes": SIZES, "middle_ops": MIDDLE_OPS, "transitions": TRANSITIONS,
                   "repeats": REPEATS, "python": sys.version.split()[0]},
    }
    print_report(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if args.plots:
        save_plots(res, args.plots)
        print(f"\nГрафики сохранены в {args.plots}/")


if __name__ == "__main__":
    main()
