
"""
Лабораторная работа № 6. Вариант 13.
Дисциплина: Теория алгоритмов.
Выполнила: Кугушева Арина Вячеславовна, группа БИ 2-1.

Тема: реализация линейных структур данных и анализ трудоёмкости их операций.

ИСХОДНЫЕ ДАННЫЕ ВАРИАНТА 13 (табл. 1 задания)
    структуры для реализации ... двусвязный список и кольцевая очередь
                                 (очередь на основе кольцевого буфера);
    исследуемые операции ....... вставка в середину, обход в обе стороны;
    размеры n .................. 1000, 2000, 4000;
    прикладная задача .......... плей-лист с переходами (вперёд / назад).

ДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ (табл. 2, для всех вариантов)
    Т1  проверка структур на малом примере с выводом содержимого после
        каждой операции;
    Т2  измерение времени каждой операции для собственной реализации и для
        встроенного list;
    Т3  графики T(n) и определение, какой оценке (Θ(1), Θ(n), Θ(n²))
        соответствует каждая кривая;
    Т4  оценка расхода памяти на 10 000 элементов (sys.getsizeof).

ЧТО ДЕЛАЕТ ПРОГРАММА (по порядку)
    0. Самопроверка: списки и очередь сравниваются со встроенными
       list и collections.deque на случайных операциях.
    1. Т1 - малый пример: содержимое структур печатается после каждой операции.
    2. Прикладная задача: плей-лист на двусвязном списке с кольцевой
       очередью «играть следующим» (сценарий с выводом).
    3. Измерения (Т2): вставка в середину, обход вперёд и назад, доступ по
       индексу, кольцевая очередь; кроме времени считается число переходов
       по ссылкам («шагов»), которое не зависит от скорости компьютера.
    4. Т3: отношения T(2n)/T(n) и показатель наклона в двойных логарифмических
       координатах, графики lr6_plot.png.
    5. Т4: расход памяти на 10 000 элементов, график lr6_plot_memory.png.

ЗАПУСК
    python lr6_variant13.py             (полный прогон, около 1 минуты)
    python lr6_variant13.py --json F    (дополнительно сохранить результаты в F)
    python lr6_variant13.py --show      (дополнительно показать графики)

Требуется: Python 3.10+, библиотека matplotlib (pip install matplotlib).
"""

import gc
import json
import math
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import deque

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # режим без окна: графики только сохраняются в файлы
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# --------------------------------------------------------------------------
SIZES = [1000, 2000, 4000]          # размеры n из варианта 13
EXT_LIST_SIZES = [10**4, 10**5, 10**6]   # дополнительные размеры для вставки в середину list
EXT_QUEUE_SIZES = [16_000, 64_000]       # дополнительные размеры для сравнения очередей
K_MID = 100                         # число вставок в середину на один замер
K_NODE = 1000                       # число вставок после известного узла на один замер
K_ACCESS = 200                      # число обращений по индексу на один замер
REPS = 7                            # число повторов замера (берётся медиана)
SEED = 13                           # начальное значение генератора случайных чисел
MEM_N = 10_000                      # число элементов для оценки памяти (Т4)


# --------------------------------------------------------------------------
# 2. ДВУСВЯЗНЫЙ СПИСОК
# --------------------------------------------------------------------------
class Node:
    """
    Узел двусвязного списка: значение и две ссылки - на предыдущий (prev)
    и следующий (next) узлы. __slots__ запрещает создание словаря атрибутов
    у каждого узла, что заметно экономит память.
    """
    __slots__ = ("value", "prev", "next")

    def __init__(self, value, prev=None, nxt=None):
        self.value = value          # хранимые данные
        self.prev = prev            # ссылка на предыдущий узел (None у головы)
        self.next = nxt             # ссылка на следующий узел (None у хвоста)


class DoublyLinkedList:
    """
    Двусвязный список со ссылками на голову (head) и хвост (tail).

    Счётчик self.hops считает число переходов по ссылкам при поиске узла по
    индексу и при «ручном» доступе с головы. Это машинно-независимая мера
    трудоёмкости, которую можно сопоставлять с теорией.

    Трудоёмкость основных операций:
        push_front, push_back, pop_front, pop_back ....... Θ(1)
        insert_after(node), remove_node(node) ............ Θ(1) - узел известен
        get(i), insert(i, v), delete(i) .................. Θ(min(i, n - i))
        find(v) .......................................... Θ(n)
        обход вперёд / назад ............................. Θ(n)
    """

    def __init__(self):
        self.head = None            # первый узел
        self.tail = None            # последний узел
        self.size = 0               # число элементов
        self.hops = 0               # счётчик переходов по ссылкам

    def __len__(self):
        return self.size

    # ---------------------------- обход ------------------------------------
    def __iter__(self):
        """Обход ВПЕРЁД от головы к хвосту по ссылкам next: Θ(n)."""
        node = self.head
        while node is not None:
            yield node.value
            node = node.next

    def reversed(self):
        """Обход НАЗАД от хвоста к голове по ссылкам prev: Θ(n)."""
        node = self.tail
        while node is not None:
            yield node.value
            node = node.prev

    def to_list(self):
        """Содержимое списка (слева направо) в виде обычного list."""
        return list(self)

    def to_list_reversed(self):
        """Содержимое списка (справа налево) в виде обычного list."""
        return list(self.reversed())

    # ------------------------ доступ по индексу -----------------------------
    def _node_at(self, index):
        """
        Возвращает узел с номером index. Благодаря двум ссылкам идём с ТОГО
        конца, который ближе: максимум n/2 переходов вместо n.
        """
        if not 0 <= index < self.size:
            raise IndexError("Индекс вне диапазона")
        if index <= self.size // 2:                 # ближе к голове
            node = self.head
            for _ in range(index):
                node = node.next
            self.hops += index
        else:                                       # ближе к хвосту
            node = self.tail
            steps = self.size - 1 - index
            for _ in range(steps):
                node = node.prev
            self.hops += steps
        return node

    def get_from_head(self, index):
        """
        Доступ по индексу ТОЛЬКО от головы (так работал бы односвязный список,
        у которого нет ссылки prev). Нужен для сравнения: обход назад через
        такие обращения стоит Θ(n²).
        """
        if not 0 <= index < self.size:
            raise IndexError("Индекс вне диапазона")
        node = self.head
        for _ in range(index):
            node = node.next
        self.hops += index
        return node.value

    def get(self, index):
        """Значение по индексу: Θ(min(i, n - i))."""
        return self._node_at(index).value

    def find(self, value):
        """Индекс первого вхождения value или -1: Θ(n)."""
        node, i = self.head, 0
        while node is not None:
            if node.value == value:
                return i
            node, i = node.next, i + 1
        return -1

    # ------------------------------ вставка ---------------------------------
    def push_front(self, value):
        """Вставка в начало: Θ(1)."""
        node = Node(value, None, self.head)         # новый узел перед головой
        if self.head is not None:
            self.head.prev = node
        else:                                       # список был пуст
            self.tail = node
        self.head = node
        self.size += 1

    def push_back(self, value):
        """Вставка в конец: Θ(1) благодаря ссылке tail."""
        node = Node(value, self.tail, None)         # новый узел после хвоста
        if self.tail is not None:
            self.tail.next = node
        else:                                       # список был пуст
            self.head = node
        self.tail = node
        self.size += 1

    def insert_after(self, node, value):
        """
        Вставка нового значения СРАЗУ ПОСЛЕ известного узла: Θ(1).
        Переставляются только четыре ссылки, сдвигать элементы не нужно.
        """
        new = Node(value, node, node.next)          # prev = node, next = node.next
        if node.next is not None:
            node.next.prev = new                    # следующий узел указывает назад на new
        else:                                       # node был хвостом
            self.tail = new
        node.next = new
        self.size += 1
        return new

    def insert(self, index, value):
        """
        Вставка значения так, чтобы оно получило номер index (0 <= index <= n).
        Для середины списка нужно дойти до позиции: Θ(n / 2) переходов,
        сама перестановка ссылок - Θ(1).
        """
        if not 0 <= index <= self.size:
            raise IndexError("Индекс вне диапазона")
        if index == 0:
            self.push_front(value)
        elif index == self.size:
            self.push_back(value)
        else:
            before = self._node_at(index - 1)       # узел, после которого вставляем
            self.insert_after(before, value)

    # ------------------------------ удаление --------------------------------
    def remove_node(self, node):
        """Удаление известного узла: Θ(1). Возвращает его значение."""
        if node.prev is not None:
            node.prev.next = node.next              # предыдущий «перепрыгивает» узел
        else:                                       # удаляем голову
            self.head = node.next
        if node.next is not None:
            node.next.prev = node.prev              # следующий ссылается на предыдущий
        else:                                       # удаляем хвост
            self.tail = node.prev
        self.size -= 1
        return node.value

    def pop_front(self):
        """Удаление из начала: Θ(1)."""
        if self.head is None:
            raise IndexError("Список пуст")
        return self.remove_node(self.head)

    def pop_back(self):
        """Удаление из конца: Θ(1) (в односвязном списке было бы Θ(n))."""
        if self.tail is None:
            raise IndexError("Список пуст")
        return self.remove_node(self.tail)

    def delete(self, index):
        """Удаление по индексу: Θ(min(i, n - i)). Возвращает значение."""
        return self.remove_node(self._node_at(index))


# --------------------------------------------------------------------------
# 3. КОЛЬЦЕВАЯ ОЧЕРЕДЬ
# --------------------------------------------------------------------------
class CircularQueue:
    """
    Очередь (FIFO) на основе КОЛЬЦЕВОГО БУФЕРА фиксированной ёмкости.

    Хранится массив из capacity ячеек, индекс головы (head) и число элементов
    (count). Индекс следующей свободной ячейки (tail) вычисляется как
        tail = (head + count) mod capacity,
    поэтому при достижении конца массива запись продолжается с его начала.
    enqueue и dequeue работают за Θ(1) и не сдвигают элементы.
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Ёмкость должна быть положительной")
        self.data = [None] * capacity   # массив-буфер
        self.capacity = capacity        # ёмкость буфера
        self.head = 0                   # индекс элемента, который выйдет первым
        self.count = 0                  # число элементов в очереди

    def __len__(self):
        return self.count

    def is_empty(self):
        return self.count == 0

    def is_full(self):
        return self.count == self.capacity

    def enqueue(self, value):
        """Постановка в очередь: Θ(1). При переполнении - OverflowError."""
        if self.is_full():
            raise OverflowError("Очередь переполнена")
        tail = (self.head + self.count) % self.capacity     # следующая свободная ячейка
        self.data[tail] = value
        self.count += 1

    def dequeue(self):
        """Извлечение из очереди: Θ(1). Из пустой очереди - IndexError."""
        if self.is_empty():
            raise IndexError("Очередь пуста")
        value = self.data[self.head]
        self.data[self.head] = None                         # освобождаем ссылку
        self.head = (self.head + 1) % self.capacity         # голова двигается по кругу
        self.count -= 1
        return value

    def front(self):
        """Первый элемент без извлечения: Θ(1)."""
        if self.is_empty():
            raise IndexError("Очередь пуста")
        return self.data[self.head]

    def to_list(self):
        """Содержимое очереди от головы к хвосту (для вывода)."""
        return [self.data[(self.head + i) % self.capacity] for i in range(self.count)]


# --------------------------------------------------------------------------
# 4. ПРИКЛАДНАЯ ЗАДАЧА: ПЛЕЙ-ЛИСТ С ПЕРЕХОДАМИ
# --------------------------------------------------------------------------
class Playlist:
    """
    Плей-лист: треки хранятся в двусвязном списке, текущий трек - ссылка
    current на узел списка.

    Переходы «следующий» и «предыдущий», вставка трека сразу после текущего
    и удаление текущего выполняются за Θ(1): позиция уже известна (это узел),
    искать по индексу не нужно. В режиме cycle=True после последнего трека
    воспроизведение продолжается с первого (и наоборот при переходе назад).

    Очередь «играть следующим» - кольцевая очередь (CircularQueue): трек,
    поставленный в неё, будет вставлен после текущего при ближайшем
    переходе вперёд, раньше остальных треков плей-листа.
    """

    def __init__(self, titles=(), cycle=True, queue_capacity=8):
        self.tracks = DoublyLinkedList()            # список треков
        for title in titles:
            self.tracks.push_back(title)
        self.current = self.tracks.head             # текущий трек (узел)
        self.cycle = cycle                          # зацикливать воспроизведение
        self.up_next = CircularQueue(queue_capacity)    # очередь «играть следующим»

    def now_playing(self):
        """Название текущего трека (или None, если плей-лист пуст)."""
        return None if self.current is None else self.current.value

    def next_track(self):
        """Переход к следующему треку по ссылке next: Θ(1)."""
        if self.current is None:
            return None
        nxt = self.current.next
        if nxt is None and self.cycle:              # дошли до конца - переходим к началу
            nxt = self.tracks.head
        if nxt is not None:
            self.current = nxt
        return self.current.value

    def prev_track(self):
        """Переход к предыдущему треку по ссылке prev: Θ(1)."""
        if self.current is None:
            return None
        prv = self.current.prev
        if prv is None and self.cycle:              # дошли до начала - переходим к концу
            prv = self.tracks.tail
        if prv is not None:
            self.current = prv
        return self.current.value

    def add_to_end(self, title):
        """Добавить трек в конец плей-листа: Θ(1)."""
        self.tracks.push_back(title)
        if self.current is None:
            self.current = self.tracks.head

    def add_after_current(self, title):
        """Вставить трек сразу после текущего: Θ(1) (в массиве потребовался бы сдвиг)."""
        if self.current is None:
            self.add_to_end(title)
        else:
            self.tracks.insert_after(self.current, title)

    def remove_current(self):
        """Удалить текущий трек и перейти к следующему (или предыдущему): Θ(1)."""
        if self.current is None:
            return None
        node = self.current
        self.current = node.next if node.next is not None else node.prev
        return self.tracks.remove_node(node)

    def queue_next(self, title):
        """Поставить трек в очередь «играть следующим»."""
        self.up_next.enqueue(title)

    def play_next(self):
        """
        Переход вперёд с учётом очереди: если в очереди «играть следующим» есть
        треки, первый из них вставляется после текущего и становится текущим;
        иначе выполняется обычный переход к следующему треку.
        """
        if not self.up_next.is_empty():
            self.add_after_current(self.up_next.dequeue())
            return self.next_track()
        return self.next_track()

    def show(self):
        """Строка вида «A | ►B | C» - текущий трек отмечен значком ►."""
        parts, node = [], self.tracks.head
        while node is not None:
            parts.append(("►" if node is self.current else "") + str(node.value))
            node = node.next
        return " | ".join(parts) if parts else "(пусто)"


# --------------------------------------------------------------------------
# 5. САМОПРОВЕРКА И МАЛЫЙ ПРИМЕР (Т1)
# --------------------------------------------------------------------------
def self_test():
    """
    Сверяет двусвязный список со встроенным list, а кольцевую очередь - с
    collections.deque на случайных операциях. При ошибке - AssertionError.
    """
    rng = random.Random(0)
    for _ in range(300):                            # много коротких случайных сценариев
        dll, ref = DoublyLinkedList(), []
        for _ in range(60):
            op = rng.choice(["pf", "pb", "ins", "popf", "popb", "del", "get"])
            v = rng.randint(0, 99)
            if op == "pf":
                dll.push_front(v)
                ref.insert(0, v)
            elif op == "pb":
                dll.push_back(v)
                ref.append(v)
            elif op == "ins":
                i = rng.randint(0, len(ref))
                dll.insert(i, v)
                ref.insert(i, v)
            elif op == "popf" and ref:
                assert dll.pop_front() == ref.pop(0)
            elif op == "popb" and ref:
                assert dll.pop_back() == ref.pop()
            elif op == "del" and ref:
                i = rng.randrange(len(ref))
                assert dll.delete(i) == ref.pop(i)
            elif op == "get" and ref:
                i = rng.randrange(len(ref))
                assert dll.get(i) == ref[i] == dll.get_from_head(i)
            # после каждой операции проверяем содержимое в обе стороны и длину
            assert dll.to_list() == ref
            assert dll.to_list_reversed() == ref[::-1]
            assert len(dll) == len(ref)
    for cap in (1, 2, 5, 16):                       # кольцевая очередь против deque
        q, ref = CircularQueue(cap), deque()
        for _ in range(500):
            if rng.random() < 0.5 and len(ref) < cap:
                v = rng.randint(0, 99)
                q.enqueue(v)
                ref.append(v)
            elif ref:
                assert q.dequeue() == ref.popleft()
            assert q.to_list() == list(ref) and len(q) == len(ref)
    print("Самопроверка пройдена (список и очередь совпадают с list и deque)")


def demo_small():
    """Т1: малый пример - содержимое структур выводится после КАЖДОЙ операции."""
    print("\n--- Т1. Малый пример: двусвязный список ---")
    dll = DoublyLinkedList()

    def show(action):
        print(f"{action:<28} вперёд: {dll.to_list()}  назад: {dll.to_list_reversed()}  длина: {len(dll)}")

    for v in (10, 20, 30, 40, 50, 60, 70, 80):
        dll.push_back(v)
    show("push_back 10..80")
    dll.push_front(5)
    show("push_front(5)")
    dll.insert(len(dll) // 2, 99)
    show(f"insert({(len(dll) - 1) // 2}, 99) - середина")
    dll.insert(1, 11)
    show("insert(1, 11)")
    mid = dll._node_at(len(dll) // 2)
    dll.insert_after(mid, 77)
    show(f"insert_after(узла {mid.value}, 77)")
    print(f"{'get(4) / find(77)':<28} {dll.get(4)} / {dll.find(77)}")
    dll.pop_front()
    show("pop_front()")
    dll.pop_back()
    show("pop_back()")
    dll.delete(3)
    show("delete(3)")

    print("\n--- Т1. Малый пример: кольцевая очередь (ёмкость 5) ---")
    q = CircularQueue(5)

    def showq(action):
        print(f"{action:<22} очередь: {q.to_list()}  head={q.head}  count={q.count}  буфер: {q.data}")

    for t in ("заявка-1", "заявка-2", "заявка-3", "заявка-4"):
        q.enqueue(t)
        showq(f"enqueue({t})")
    print(f"{'dequeue()':<22} -> {q.dequeue()}")
    showq("после dequeue")
    print(f"{'dequeue()':<22} -> {q.dequeue()}")
    showq("после dequeue")
    for t in ("заявка-5", "заявка-6", "заявка-7"):   # tail переходит через конец массива
        q.enqueue(t)
        showq(f"enqueue({t})")
    try:
        q.enqueue("заявка-8")
    except OverflowError as e:
        print(f"{'enqueue(заявка-8)':<22} -> {e}")


def demo_playlist():
    """Прикладная задача: сценарий работы с плей-листом (вывод после каждого шага)."""
    print("\n--- Прикладная задача: плей-лист с переходами ---")
    pl = Playlist(["Трек A", "Трек B", "Трек C", "Трек D", "Трек E"])

    def step(action, result=None):
        suffix = "" if result is None else f"  -> играет: {result}"
        print(f"{action:<34} {pl.show()}{suffix}")

    step("начальное состояние")
    step("next_track()", pl.next_track())
    step("next_track()", pl.next_track())
    step("prev_track()", pl.prev_track())
    pl.add_after_current("Трек X")
    step("add_after_current('Трек X')")
    pl.queue_next("Трек Y")
    pl.queue_next("Трек Z")
    print(f"{'queue_next('+chr(39)+'Трек Y'+chr(39)+'), Z':<34} очередь: {pl.up_next.to_list()}")
    step("play_next() (из очереди)", pl.play_next())
    step("play_next() (из очереди)", pl.play_next())
    step("play_next()", pl.play_next())
    pl.remove_current()
    step("remove_current()", pl.now_playing())
    for _ in range(3):
        pl.next_track()
    step("3 x next_track() - конец", pl.now_playing())
    step("next_track() - к началу (цикл)", pl.next_track())
    step("prev_track() - к концу (цикл)", pl.prev_track())


# --------------------------------------------------------------------------
# 6. ИЗМЕРЕНИЯ (Т2)
# --------------------------------------------------------------------------
def median_time(run, reps=REPS):
    """
    Медиана по reps повторам. run() сама измеряет нужный участок и
    возвращает время в секундах (подготовка данных в замер не входит).
    """
    run()                                           # «прогревочный» запуск: его результат отбрасывается
    return statistics.median(run() for _ in range(reps))


def timed(func):
    """
    Время выполнения func() в секундах. На время замера выключается сборщик
    мусора, чтобы его случайные срабатывания не искажали измерения.
    """
    gc.collect()                                    # заранее убираем накопившийся мусор
    gc.disable()
    try:
        t0 = time.perf_counter()
        func()
        return time.perf_counter() - t0
    finally:
        gc.enable()


def build_dll(n):
    """Двусвязный список из чисел 0..n-1."""
    dll = DoublyLinkedList()
    for i in range(n):
        dll.push_back(i)
    return dll


def measure_mid_insert(n):
    """Вставка в середину: время ОДНОЙ вставки (DLL по индексу, DLL после узла, list)."""
    def run_dll():
        dll = build_dll(n)

        def work():
            for _ in range(K_MID):
                dll.insert(len(dll) // 2, 0)
        return timed(work)

    def run_node():
        dll = build_dll(n)
        node = dll._node_at(n // 2)                 # ссылка на средний узел уже есть

        def work():
            for _ in range(K_NODE):
                dll.insert_after(node, 0)
        return timed(work)

    def run_list():
        lst = list(range(n))

        def work():
            for _ in range(K_MID * 10):             # у list операция быстрая - берём больше повторов
                lst.insert(len(lst) // 2, 0)
        return timed(work)

    dll = build_dll(n)                              # отдельный прогон для счётчика шагов
    for _ in range(K_MID):
        dll.insert(len(dll) // 2, 0)
    return {
        "dll_index": median_time(run_dll) / K_MID,
        "dll_node": median_time(run_node) / K_NODE,
        "list": median_time(run_list) / (K_MID * 10),
        "hops_per_op": dll.hops / K_MID,
    }


def measure_traversal(n):
    """Обход всех n элементов: вперёд, назад, «назад через индексы от головы» и list."""
    dll = build_dll(n)
    lst = list(range(n))

    def run_fwd():
        return timed(lambda: [None for _ in dll])

    def run_back():
        return timed(lambda: [None for _ in dll.reversed()])

    def run_naive():                                # так выглядел бы обход назад без ссылок prev
        def work():
            for i in range(n - 1, -1, -1):
                dll.get_from_head(i)
        return timed(work)

    def run_list_fwd():
        return timed(lambda: [None for _ in lst])

    def run_list_back():
        return timed(lambda: [None for _ in reversed(lst)])

    dll.hops = 0
    for i in range(n - 1, -1, -1):
        dll.get_from_head(i)
    naive_hops = dll.hops                           # фиксируем сразу: сумма 0 + 1 + ... + (n - 1)
    return {
        "fwd": median_time(run_fwd),
        "back": median_time(run_back),
        "naive_back": median_time(run_naive, reps=3),
        "list_fwd": median_time(run_list_fwd),
        "list_back": median_time(run_list_back),
        "naive_hops": naive_hops,
    }


def measure_access(n):
    """Доступ по индексу: время одного обращения к случайной позиции (DLL и list)."""
    rng = random.Random(SEED + n)
    idx = [rng.randrange(n) for _ in range(K_ACCESS)]
    dll = build_dll(n)
    lst = list(range(n))
    dll.hops = 0
    for i in idx:
        dll.get(i)
    hops = dll.hops / K_ACCESS
    return {
        "dll": median_time(lambda: timed(lambda: [dll.get(i) for i in idx])) / K_ACCESS,
        "list": median_time(lambda: timed(lambda: [lst[i] for i in idx])) / K_ACCESS,
        "hops_per_op": hops,
    }


def measure_queue(n):
    """Очередь: n постановок и n извлечений (суммарное время), кольцо / list.pop(0) / deque."""
    def run_ring():
        q = CircularQueue(n)

        def work():
            for i in range(n):
                q.enqueue(i)
            for _ in range(n):
                q.dequeue()
        return timed(work)

    def run_list():
        lst = []

        def work():
            for i in range(n):
                lst.append(i)
            for _ in range(n):
                lst.pop(0)                          # извлечение из начала - сдвиг, Θ(n)
        return timed(work)

    def run_deque():
        d = deque()

        def work():
            for i in range(n):
                d.append(i)
            for _ in range(n):
                d.popleft()
        return timed(work)

    return {"ring": median_time(run_ring, 3), "list_pop0": median_time(run_list, 3),
            "deque": median_time(run_deque, 3)}


def measure_list_mid_ext(n):
    """Время одной вставки в середину list для больших n (подтверждение Θ(n))."""
    lst = list(range(n))
    reps = 200

    def work():
        for _ in range(reps):
            lst.insert(len(lst) // 2, 0)
    return statistics.median(timed(work) for _ in range(3)) / reps


def slope(sizes, times):
    """Показатель k в T ~ n^k: наклон линейной регрессии log T от log n."""
    xs = [math.log(s) for s in sizes]
    ys = [math.log(t) for t in times]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


# --------------------------------------------------------------------------
# 7. ПАМЯТЬ (Т4)
# --------------------------------------------------------------------------
def deep_size_list(lst):
    """Размер list вместе с объектами int внутри (sys.getsizeof не «глубокий»)."""
    return sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)


def deep_size_dll(dll):
    """Размер двусвязного списка: все узлы Node плюс объекты int."""
    total, node = sys.getsizeof(dll), dll.head
    while node is not None:
        total += sys.getsizeof(node) + sys.getsizeof(node.value)
        node = node.next
    return total


def deep_size_ring(q):
    """Размер кольцевой очереди: массив-буфер плюс объекты int."""
    return sys.getsizeof(q) + sys.getsizeof(q.data) + sum(
        sys.getsizeof(x) for x in q.data if x is not None)


def peak_memory(builder):
    """Память (байты), занятая объектом, который строит builder(): по tracemalloc."""
    tracemalloc.start()
    obj = builder()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return current


def experiment_memory():
    """Т4: расход памяти на MEM_N элементов для list, двусвязного списка и очереди."""
    n = MEM_N
    values = list(range(n))
    lst = list(values)
    dll = build_dll(n)
    q = CircularQueue(n)
    for v in values:
        q.enqueue(v)
    ints = sum(sys.getsizeof(x) for x in values)

    def build_q():
        qq = CircularQueue(n)
        for v in range(n):
            qq.enqueue(v)
        return qq

    return {
        "n": n,
        "int_size": sys.getsizeof(12345),
        "node_size": sys.getsizeof(dll.head),
        "list_getsizeof": deep_size_list(lst),
        "dll_getsizeof": deep_size_dll(dll),
        "ring_getsizeof": deep_size_ring(q),
        "ints_total": ints,
        "list_tracemalloc": peak_memory(lambda: list(range(n))),
        "dll_tracemalloc": peak_memory(lambda: build_dll(n)),
        "ring_tracemalloc": peak_memory(build_q),
    }


# --------------------------------------------------------------------------
# 8. ГРАФИКИ
# --------------------------------------------------------------------------
def make_plots(res):
    """Строит и сохраняет рисунки lr6_plot.png и lr6_plot_memory.png."""
    sizes = SIZES
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0][0]                                 # вставка в середину
    mi = res["mid_insert"]
    ax.plot(sizes, [mi[str(n)]["dll_index"] * 1e6 for n in sizes], "o-", label="DLL: insert(i, v)")
    ax.plot(sizes, [mi[str(n)]["dll_node"] * 1e6 for n in sizes], "s-", label="DLL: insert_after(узел)")
    ax.plot(sizes, [mi[str(n)]["list"] * 1e6 for n in sizes], "^-", label="list.insert(i, v)")
    ax.set_title("Вставка в середину (одна операция)")
    ax.set_ylabel("мкс")

    ax = axes[0][1]                                 # обход
    tr = res["traversal"]
    ax.plot(sizes, [tr[str(n)]["fwd"] * 1e3 for n in sizes], "o-", label="DLL вперёд (next)")
    ax.plot(sizes, [tr[str(n)]["back"] * 1e3 for n in sizes], "s-", label="DLL назад (prev)")
    ax.plot(sizes, [tr[str(n)]["naive_back"] * 1e3 for n in sizes], "d-", label="назад по индексам от головы")
    ax.plot(sizes, [tr[str(n)]["list_fwd"] * 1e3 for n in sizes], "^-", label="list вперёд")
    ax.set_title("Обход всех n элементов")
    ax.set_ylabel("мс")

    ax = axes[1][0]                                 # доступ по индексу
    ac = res["access"]
    ax.plot(sizes, [ac[str(n)]["dll"] * 1e6 for n in sizes], "o-", label="DLL: get(i)")
    ax.plot(sizes, [ac[str(n)]["list"] * 1e6 for n in sizes], "^-", label="list[i]")
    ax.set_title("Доступ по индексу (одно обращение)")
    ax.set_ylabel("мкс")

    ax = axes[1][1]                                 # очередь
    qs = sorted(int(k) for k in res["queue"])
    ax.plot(qs, [res["queue"][str(n)]["ring"] * 1e3 for n in qs], "o-", label="кольцевая очередь")
    ax.plot(qs, [res["queue"][str(n)]["deque"] * 1e3 for n in qs], "s-", label="collections.deque")
    ax.plot(qs, [res["queue"][str(n)]["list_pop0"] * 1e3 for n in qs], "^-", label="list.pop(0)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Очередь: n постановок + n извлечений")
    ax.set_ylabel("мс (лог. шкала)")

    for ax in axes.flat:
        ax.set_xlabel("n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lr6_plot.png", dpi=150)
    plt.close(fig)

    # --- память -------------------------------------------------------------
    m = res["memory"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["list", "Двусвязный\nсписок", "Кольцевая\nочередь"]
    vals = [m["list_getsizeof"], m["dll_getsizeof"], m["ring_getsizeof"]]
    bars = ax.bar(labels, [v / 1024 for v in vals], color=["tab:blue", "tab:orange", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v / m['n']:.0f} Б/эл.",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("КиБ (sys.getsizeof, глубокий подсчёт)")
    ax.set_title(f"Память на {m['n']:,} элементов".replace(",", " "))
    fig.tight_layout()
    fig.savefig("lr6_plot_memory.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 9. ВЫВОД РЕЗУЛЬТАТОВ
# --------------------------------------------------------------------------
def system_info():
    """Строка с характеристиками системы (вставляется в отчёт)."""
    cpu = platform.processor() or platform.machine()
    try:                                            # на Linux модель процессора лежит в /proc/cpuinfo
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"ОС: {platform.system()} {platform.release()}; процессор: {cpu}; "
            f"Python {platform.python_version()}")


def print_report(res):
    """Печатает результаты измерений в консоль."""
    mi, tr, ac, qu = res["mid_insert"], res["traversal"], res["access"], res["queue"]
    print("\nТаблица 1. Вставка в середину: время одной операции, мкс (и число переходов по ссылкам)")
    print(f"  {'n':>6}{'DLL insert(i)':>16}{'переходов':>11}{'DLL после узла':>16}{'list.insert':>14}")
    for n in SIZES:
        r = mi[str(n)]
        print(f"  {n:>6}{r['dll_index'] * 1e6:>16.2f}{r['hops_per_op']:>11.0f}"
              f"{r['dll_node'] * 1e6:>16.3f}{r['list'] * 1e6:>14.3f}")
    print("  list.insert в середину для больших n (мкс):",
          ", ".join(f"n={n}: {t * 1e6:.1f}" for n, t in res["list_mid_ext"].items()))

    print("\nТаблица 2. Обход всех n элементов, мс")
    print(f"  {'n':>6}{'DLL вперёд':>12}{'DLL назад':>12}{'назад от головы':>17}{'переходов':>12}"
          f"{'list вперёд':>13}{'list назад':>12}")
    for n in SIZES:
        r = tr[str(n)]
        print(f"  {n:>6}{r['fwd'] * 1e3:>12.3f}{r['back'] * 1e3:>12.3f}{r['naive_back'] * 1e3:>17.2f}"
              f"{r['naive_hops']:>12}{r['list_fwd'] * 1e3:>13.3f}{r['list_back'] * 1e3:>12.3f}")

    print("\nТаблица 3. Доступ по индексу: время одного обращения, мкс (и переходов)")
    print(f"  {'n':>6}{'DLL get(i)':>12}{'переходов':>11}{'list[i]':>10}")
    for n in SIZES:
        r = ac[str(n)]
        print(f"  {n:>6}{r['dll'] * 1e6:>12.3f}{r['hops_per_op']:>11.1f}{r['list'] * 1e6:>10.4f}")

    print("\nТаблица 4. Очередь: n постановок + n извлечений, мс")
    print(f"  {'n':>7}{'кольцевая':>12}{'deque':>10}{'list.pop(0)':>14}")
    for n in sorted(int(k) for k in qu):
        r = qu[str(n)]
        print(f"  {n:>7}{r['ring'] * 1e3:>12.2f}{r['deque'] * 1e3:>10.2f}{r['list_pop0'] * 1e3:>14.2f}")

    print("\nТаблица 5. Т3: отношение T(2n)/T(n) и показатель k в T ~ n^k")
    for name, ser in res["scaling"].items():
        print(f"  {name:<44} T(2n)/T(n): {', '.join(f'{x:.2f}' for x in ser['ratios'])};  k = {ser['k']:.2f}")

    m = res["memory"]
    print(f"\nТаблица 6. Т4: память на {m['n']} элементов (int = {m['int_size']} Б, узел Node = {m['node_size']} Б)")
    for key, name in (("list", "list"), ("dll", "двусвязный список"), ("ring", "кольцевая очередь")):
        g, t = m[f"{key}_getsizeof"], m[f"{key}_tracemalloc"]
        print(f"  {name:<20} getsizeof: {g:>9} Б ({g / m['n']:.1f} Б/эл.);  tracemalloc: {t:>9} Б ({t / m['n']:.1f} Б/эл.)")


def make_scaling(res):
    """Т3: для каждой серии измерений считает T(2n)/T(n) и показатель наклона k."""
    mi, tr, ac = res["mid_insert"], res["traversal"], res["access"]
    series = {
        "DLL: вставка в середину по индексу": [mi[str(n)]["dll_index"] for n in SIZES],
        "DLL: вставка после известного узла": [mi[str(n)]["dll_node"] for n in SIZES],
        "list.insert в середину": [mi[str(n)]["list"] for n in SIZES],
        "DLL: обход вперёд": [tr[str(n)]["fwd"] for n in SIZES],
        "DLL: обход назад (prev)": [tr[str(n)]["back"] for n in SIZES],
        "Обход назад по индексам от головы": [tr[str(n)]["naive_back"] for n in SIZES],
        "list: обход вперёд": [tr[str(n)]["list_fwd"] for n in SIZES],
        "DLL: доступ по индексу": [ac[str(n)]["dll"] for n in SIZES],
        "list[i]": [ac[str(n)]["list"] for n in SIZES],
        "Кольцевая очередь (n + n операций)": [res["queue"][str(n)]["ring"] for n in SIZES],
    }
    out = {}
    for name, times in series.items():
        out[name] = {"ratios": [times[i + 1] / times[i] for i in range(len(times) - 1)],
                     "k": slope(SIZES, times)}
    return out


def main():
    print("Лабораторная работа № 6, вариант 13")
    print(system_info())
    print(f"Структуры: двусвязный список, кольцевая очередь; n = {SIZES}; повторов: {REPS} (медиана)")
    self_test()
    demo_small()
    demo_playlist()

    res = {}
    print("\nИзмерения (Т2) ...", flush=True)
    res["mid_insert"] = {str(n): measure_mid_insert(n) for n in SIZES}
    res["list_mid_ext"] = {str(n): measure_list_mid_ext(n) for n in EXT_LIST_SIZES}
    res["traversal"] = {str(n): measure_traversal(n) for n in SIZES}
    res["access"] = {str(n): measure_access(n) for n in SIZES}
    res["queue"] = {str(n): measure_queue(n) for n in SIZES + EXT_QUEUE_SIZES}
    res["scaling"] = make_scaling(res)
    res["memory"] = experiment_memory()
    print_report(res)
    make_plots(res)
    print("\nГрафики сохранены: lr6_plot.png, lr6_plot_memory.png")
    if "--json" in sys.argv:                        # служебный вывод для отчёта
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
