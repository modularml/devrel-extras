# Hands-on with Mojo 24.6

Mojo 24.6 has arrived with significant changes to argument conventions and lifetime management. This release marks an important step in Mojo's evolution, making its memory and ownership model more intuitive while maintaining strong safety guarantees. The changes are available now through the Magic package manager and are bundled with the [MAX 24.6 release](https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform).

In this blog post, we'll explore these changes through practical examples that demonstrate the new syntax and features. We'll start with basic argument conventions and gradually introduce more advanced concepts like origins (formerly lifetimes) and implicit conversions. By the end, you'll have a thorough understanding of how to use these enhancements in your Mojo code.

One of the biggest highlights of this release is the significant contributions from our community. We received numerous pull requests from 11 community contributors that included new features, bug fixes, documentation enhancements, and code refactoring.

Special thanks to our community contributors: @jjvraw, @artemiogr97, @martinvuyk, @jayzhan211, @bgreni, @mzaks, @msaelices, @rd4com, @jiex-liu, @kszucs, @thatstoasty

For a complete list of changes, please refer to the [changelog for version 24.6](https://docs.modular.com/mojo/changelog#v246-2024-12-17).
All the code for this blog post is available in our [GitHub repository](https://github.com/modularml/devrel-extras/tree/main/blogs/hands-on-with-mojo-24-6).

## Key changes overview

One of the highlights of this release is the renaming of several core concepts to better reflect their purpose:

- `inout` → `mut` for mutable arguments: This better reflects that these parameters can be modified
- `borrowed` → `read` for read-only arguments: More clearly indicates the read-only nature of these parameters
- "lifetime" → "origin" for reference tracking: Better describes the concept of where references come from

These naming changes make Mojo code more intuitive while preserving the strong safety guarantees that developers expect. Let's dive into each feature with practical examples.

## New argument conventions

The most visible change in Mojo 24.6 is the renaming of argument conventions. The `inout` and `borrowed` keywords have been replaced with `mut` and `read` respectively. This change makes the code's intent clearer - `mut` explicitly indicates that a parameter can be modified, while `read` clearly shows that a parameter is for reading only.

Let's look at a practical example:

## Language changes

The language changes in 24.6 are

- **New Argument Conventions:** The `inout` and `borrowed` argument conventions have been renamed to `mut` and `read` respectively, better reflecting their actual purpose

- **Constructor Changes:** The `self` argument in constructors now uses the new `out` convention instead of `inout`, properly reflecting that constructors initialize without reading from `self`.

Before 24.6

```mojo
@value
struct Task:
    var description: String

    fn __init__(inout self, desc: String):
        self.description = desc

struct TaskManager:
    var tasks: List[Task]

    fn __init__(inout self):
        self.tasks = List[Task]()

    fn add_task(inout self, task: Task):
        self.tasks.append(task)

    fn show_tasks(borrowed self):
        for t in self.tasks:
            print("- ", t[].description)
```

After 24.6

```mojo
@value
struct Task:
    var description: String

    # Notice: `inout` -> `out`
    fn __init__(out self, desc: String):
        # 'out self' indicates we are constructing a fresh Task
        self.description = desc

struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        # Another 'out self' constructor, sets up an empty list of tasks
        self.tasks = List[Task]()

    # Notice: `inout` -> `mut`
    fn add_task(mut self, task: Task):
        self.tasks.append(task)

    # Notice: `borrowed` -> `read` which is also the default
    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)


def main():
    # Create a new TaskManager
    manager = TaskManager()

    # Add tasks
    manager.add_task(Task("Walk the dog"))
    manager.add_task(Task("Write Mojo 24.6 blog post"))

    manager.show_tasks()
```

To run the example with `magic` CLI:

```bash
git clone https://github.com/modularml/devrel-extras
cd devrel-extras/blogs/hands-on-with-mojo-24-6
magic run example1
```

## Implicit conversions

Another important change in 24.6 is how implicit conversions are handled. Single-argument constructors now require the `@implicit` decorator to allow implicit conversions. This makes type conversions more explicit and safer by requiring developers to opt-in to implicit conversion behavior.

Here's how it works:

```mojo
struct Task:
    var description: String

    @implicit  # Explicitly opt-in to implicit conversion
    fn __init__(out self, desc: String):
        self.description = desc

    @implicit  # Also allow conversion from string literals
    fn __init__(out self, desc: StringLiteral):
        self.description = desc
```

This change allows for more convenient usage while maintaining type safety:

```mojo
def main():
    manager = TaskManager()

    # These now work because we opted into implicit conversion
    manager.add_task("Walk the dog")  # StringLiteral → Task
    manager.add_task("Write Mojo 24.6 blog post")  # StringLiteral → Task
```

Without the `@implicit` decorator, you would need to explicitly create Task objects:

```mojo
manager.add_task(Task("Walk the dog"))  # Explicit conversion required
```

This new approach strikes a balance between convenience and type safety by making implicit conversions opt-in rather than automatic.

```bash
magic run example2
```

## Origins: a more intuitive reference model

One of the most significant changes in Mojo 24.6 is renaming "lifetimes" to "origins". This change better reflects what these annotations actually do - tracking where references come from rather than their complete lifecycle.

Let's explore this concept with some practical examples:

```mojo
struct TaskManager:
    var tasks: List[Task]

    fn get_task(ref self, index: Int) -> ref [self.tasks] Task:
        # The [self.tasks] annotation shows this reference originates from the tasks list
        return self.tasks[index]

def main():
    manager = TaskManager()
    manager.add_task("Initial task")

    # Get a reference to the first task
    first_task = manager.get_task(0)
    first_task.description = "Modified task"  # Safe modification
```

The origin annotation `[self.tasks]` clearly indicates that the returned reference comes from the tasks list.
This makes it easier to understand and track reference relationships in your code.

```bash
magic run example3
```

### Working with multiple origins

Origins become particularly powerful when working with multiple references:

```mojo
fn pick_longer(ref t1: Task, ref t2: Task) -> ref [t1, t2] Task:
    # The [t1, t2] annotation shows this reference could come from either t1 or t2
    return t1 if len(t1.description) >= len(t2.description) else t2

def main():
    manager = TaskManager()
    manager.add_task("Short task")
    manager.add_task("This is a longer task")

    # Compare tasks by length
    longer = pick_longer(
        manager.get_task(0),
        manager.get_task(1)
    )
    print("Longer task:", longer.description)
```

```bash
magic run example4
```

## Named result with `out`

Mojo 24.6 introduces a simpler syntax for named results by using the `out` convention directly. This replaces the previous `Type as out` syntax, making it more consistent with how we use `out` elsewhere in the language.

Let's look at an example that demonstrates this new syntax:

```mojo
struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    # Named result using the new 'out' convention
    @staticmethod
    fn bootstrap_example(out manager: TaskManager):
        manager = TaskManager()
        manager.add_task("Default Task #1")
        manager.add_task("Default Task #2")
        return  # 'manager' is implicitly returned

def main():
    # Create a TaskManager with default tasks
    mgr = TaskManager.bootstrap_example()
    mgr.show_tasks()
```

The `bootstrap_example` method demonstrates how named results work:

  1. We declare the result parameter with `out manager: TaskManager`
  2. We initialize and populate the manager inside the function
  3. The `return` statement implicitly returns the named result

Try this example:

```bash
magic run example5
```

## New collection types

Mojo 24.6 introduces two important additions to the standard library: `Deque` and `OwnedPointer`. Let's explore each one:

### Deque: double-ended queue

The new `Deque` collection type provides efficient operations at both ends of the sequence. This is particularly useful when you need to add or remove elements from either end of a collection, such as implementing a work queue with priorities.

```mojo
from collections import Deque

struct TaskManager:
    var tasks: Deque[Task]  # Using Deque instead of List

    fn __init__(out self):
        self.tasks = Deque[Task]()

    fn add_task(mut self, task: Task):
        self.tasks.append(task)  # Add to back (normal priority)

    fn add_urgent_task(mut self, task: Task):
        self.tasks.appendleft(task)  # Add to front (high priority)
```

Let's try this example:

```bash
magic run example6
```

### OwnedPointer: safe memory management

The `OwnedPointer` type provides safe, single-owner, non-nullable smart pointer functionality. This is particularly useful when dealing with resources that need deterministic cleanup, such as file handles or network connections.

Key features of `OwnedPointer`:

- Single ownership semantics
- Automatic cleanup when going out of scope
- Move semantics with the `^` operator
- Non-nullable (always points to valid data)

Here's a basic example showing the concept:

```mojo
from memory import OwnedPointer

@value
struct HeavyResource:
    var data: String

    fn __init__(out self, data: String):
        self.data = data

    fn do_work(read self):
        print("Processing:", self.data)

struct Task:
    var description: String
    var resource: HeavyResource

    fn __init__(out self, desc: StringLiteral):
        self.description = desc
        self.resource = HeavyResource("Resource for: " + desc)

    fn __moveinit__(out self, owned other: Task):
        self.description = other.description^
        self.resource = other.resource^
```

Try the complete example:

```mojo
from memory import OwnedPointer

@value
struct HeavyResource:
    var data: String

    fn __init__(out self, data: String):
        self.data = data

    fn do_work(read self):
        print("Heavy work:", self.data)

struct Task:
    var description: String
    var heavy_resource: HeavyResource

    @implicit
    fn __init__(out self, desc: StringLiteral):
        self.description = desc
        self.heavy_resource = HeavyResource("Heavy resource with description: " + desc)

    fn __moveinit__(out self, owned other: Task):
        self.description = other.description^
        self.heavy_resource = other.heavy_resource^

struct TaskManager:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    # changing `read` to `owned` to avoid Copy
    fn add_task(mut self, owned task: Task):
        self.tasks.append(task^)

    fn show_tasks(read self):
        for t in self.tasks:
            print("- ", t[].description)

    @staticmethod
    fn bootstrap_example(out manager: TaskManager):
        manager = TaskManager()
        manager.add_task("Resourceful Task A")
        manager.add_task("Resourceful Task B")
        return

    fn do_work(owned self):
        for t in self.tasks:
            t[].do_work()

def main():
    manager = TaskManager()
    manager.add_task("Important Task")
    manager.add_task("Another Task")

    print("Tasks:")
    mgr.show_tasks()
    print("Do work:")
    mgr^.do_work()
    # When 'mgr' goes out of scope, OwnedPointer cleans up all HeavyResources
    # so no need for explicit cleanup

```

Run it with:

```bash
magic run example7
```

## Putting it all together

Let's look at a complete example that combines all these new features. This example demonstrates how the various improvements in Mojo 24.6 work together to create cleaner, safer code:

- Using `mut`, `read`, and `out` conventions for clear intent
- Leveraging `@implicit` for convenient conversions
- Managing resources with move semantics
- Using the new `Deque` collection type
- Proper cleanup of resources through ownership rules
