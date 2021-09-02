"""template for having K parallel workers that periodically push results to a master thread"""
from ray.util.queue import Empty, Queue
import ray
import time



# If you get the following error:
# Traceback(most recent call last):
#   File "[..]/ray/actor.py", line 823, in __del__
# AttributeError: 'NoneType' object has no attribute 'global_worker'
# You need to apply this patch: https://github.com/ray-project/ray/pull/16955/commits/faa8d6b200f52d86fa5ace2506739d88a767c03b
ray.init(ignore_reinit_error=True)

@ray.remote
class Producer:
    def __init__(self, create_generator, queue, batch_size=1) -> None:
        self.create_generator = create_generator
        self.queue = queue
        self.batch_size = batch_size

    def run(self):
        gen = self.create_generator()
        items = []
        n = 0
        total = 0
        for el in gen:
            items.append(el)
            n += 1
            total += 1
            if n == self.batch_size:
                self.queue.put_nowait_batch(items)
                items = []
                n = 0
        if items:
            self.queue.put_nowait_batch(items)
        return total

@ray.remote
class Filter:
    def __init__(self, filter, from_queue, to_queue, batch_size=1, sleep_delay=.1, stop_number=1200) -> None:
        self.filter = filter
        self.from_queue: Queue = from_queue
        self.to_queue: Queue = to_queue
        self.batch_size = batch_size
        self.sleep_delay = sleep_delay
        self.stop_number = stop_number
        
    def run(self):
        n = 0
        times_without_new_data = 0
        while times_without_new_data < self.stop_number:
            try:
                items = self.from_queue.get_nowait_batch(self.batch_size)
                times_without_new_data = 0
                for item in items:
                    n += 1
                    if self.filter(item):
                        self.to_queue.put(item)
            except Empty:
                times_without_new_data += 1
                time.sleep(self.sleep_delay)     
        return n


def make_parallel_pipelines(create_generators, make_filter_fn, nb_filters, 
                            generation_queue_size, output_queue_size, batch_size,
                            filters_check_delay=.1, filters_stop_number=1200):
    """
    Parameters:
    -----------
    - create_generators: list[unit -> Generator['a]]  - functions that instanciate the generators used by producers, 1 generator = 1 producer
    - make_filter_fn: unit -> 'a -> bool              - create the filter function f: 'a -> bool, if f(a) the filter add this item to the output queue otherwise discards it 
    - nb_filters: int                                 - the number of filters actors to make
    - generation_queue_size: int                      - the max_size of the queue used to communicate between the producers and the filters
    - output_queue_size: int                          - the max_size of the queue the filters output to
    - batch_size: int                                 - producers and filters push and fetch from the queue in batches of batch_size
    - filters_check_delay: float (seconds)            - when a filter can't fetch from the queue it sleeps for this delay
    - filters_stop_number: int                        - number of consecutive times the filter must fail to fetch before the filter is stopped

    Return:
    -------
    - producers           - they need to be started with start(producers)
    - filters             - they need to be started with start(filters)
    - from_queue: Queue   - the queue used to communicate between the producers and the filters 
    - to_queue: Queue     - the output queue where the filters output to
    """
    from_queue = Queue(generation_queue_size)
    to_queue = Queue(output_queue_size)
    
    producers = [Producer.remote(
        cr, from_queue, batch_size) for cr in create_generators]
    filters = [Filter.remote(ray.put(make_filter_fn()), from_queue, to_queue, batch_size, filters_check_delay, filters_stop_number)
               for _ in range(nb_filters)]

    return producers, filters, from_queue, to_queue
        

def start(producers):
    """
    Take a list of actors and call their "run" function remotely (as in ray so in parallel).
    The returned object can be used to wait for the completion of the tasks with ray.get.
    """
    return [producer.run.remote() for producer in producers]    


if __name__ == "__main__":
    def make_generator(k):
        return lambda: (x for x in range(k))
    make_consume_fn = lambda: lambda x: x == 7
    create_generators = [make_generator(k) for k in range(10)]
    producers, filters, from_queue, to_queue = make_parallel_pipelines(
        create_generators, make_consume_fn, 2, 100, 100, 1)
    waitable_producers = start(producers)
    print("Producers have started.")
    waitable_filters = start(filters)
    print("Filters have started.")
    ray.get(waitable_producers)
    print("Producers have terminated.")
    ray.get(waitable_filters)
    print("Filters have terminated.")
    from_queue.shutdown()
    print(to_queue.size())
    to_queue.shutdown()
