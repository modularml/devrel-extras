from collections import Deque
from memory import OwnedPointer
from os import abort

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

    # We keep the @implicit for description
    @implicit
    fn __init__(out self, desc: StringLiteral):
        self.description = desc
        self.heavy_resource = HeavyResource("Heavy resource with description: " + desc)

    fn __moveinit__(out self, owned other: Task):
        self.description = other.description^
        self.heavy_resource = other.heavy_resource^

    # Workaround `CollectionElement` requirement for `Deque`
    fn __copyinit__(out self, read other: Task):
        abort("__copyinit__ should never be called")
        while True:
            pass

    fn do_work(read self):
        self.heavy_resource.do_work()


struct TaskManager:
    var tasks: Deque[Task]

    fn __init__(out self):
        self.tasks = Deque[Task]()

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
    mgr = TaskManager.bootstrap_example()
    print("Tasks:")
    mgr.show_tasks()
    print("Do work:")
    mgr^.do_work()
    # When 'mgr' goes out of scope, OwnedPointer cleans up all HeavyResources
    # so no need for explicit cleanup
