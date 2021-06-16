import operator
import random
import threading
import queue
import tempfile
import os
import time
import sys
import atexit
import numpy

import settings
from Lagrange import Lagrange

class Job:
    def __init__(self, func):
        self.ready = False
        self.func = func
        self.res = None

    def calculate(self, shares):
        self.res = self.func(shares)
        self.ready = True

    def get_res(self):
        return self.res

    def is_ready(self):
        return self.ready


# TODO: Shares should be set a job as well
class MyThread (threading.Thread):
    def __init__(self, thread_id, name, finish_queue, shares):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.shares = shares
        self.jobs = queue.Queue()
        self.finish_queue = finish_queue
        self.exit = False

    def run(self):
        while not self.exit:
            try:
                job = self.jobs.get(True, 1)
                job.calculate(self.shares)
                self.finish_queue.put(job)
            except queue.Empty:
                pass

    def stop_thread(self):
        self.exit = True

    def add_job(self, job):
        self.jobs.put(job)


class MultiPartyMediator:

    def __init__(self):
        self.counter = 0
        self.bytes = 0
        self.ready_jobs = queue.Queue()
        self.threads = []
        self.threads_created = False
        self.shares = []

    def create_threads(self):
        if self.threads_created:
            return

        self.threads_created = True
        self.threads = [0] * settings.num_threads
        if len(self.threads) > 1:
            for i in range(0, settings.num_threads):
                self.threads[i] = MyThread(i, "Thread", self.ready_jobs, self.shares[i])
                self.threads[i].start()

        atexit.register(self.cleanup)

    def cleanup(self):
        if len(self.threads) > 1:
            for i in range(0, len(self.threads)):
                self.threads[i].stop_thread()
                
            for i in range(0, len(self.threads)):
                self.threads[i].join()

        self.threads = []

    # Calculate value of x
    def start(self, n):
        #f = open("C:/Users/kost/PycharmProjects/projectGit/print_output.txt", 'w')
        #sys.stdout = f  # Change the standard output to the file we created.

        # create secret shares
        add_values = [random.randint(0, 10) for x in range(n)]
        lagranges = [Lagrange(n) for n in add_values]
        self.shares = [lag.secretShare(settings.num_threads) for lag in lagranges]
        self.shares = numpy.array(self.shares).transpose()

        self.create_threads()
        start = time.time()
        self.counter += 1
        res = 0
        
        jobs = []
        for i in range(0, settings.num_threads):
            job = Job( lambda arr: arr[0] + arr[1] )
            self.bytes += sys.getsizeof(job)
            jobs.append(job)
            self.threads[i].add_job(job)

        finished = False
        ready = 0
        while not finished:
            ready_job = self.ready_jobs.get()
            if ready_job in jobs:
                ready += 1
                self.bytes += sys.getsizeof(ready_job)
            else:
                self.ready_jobs.put(ready_job)

            if ready == len(jobs):
                finished = True

        # Now, lets calculate the result
        results = [job.get_res() for job in jobs]
        res = Lagrange.interpolate(results)

        end = time.time()
        print("Calculated: ", add_values[0], " + ", add_values[1]," = ", res)
        print("Finished calculation in ", end - start, " seconds")

        return res

    def get_counter(self):
        return self.counter

    def get_bytes(self):
        return self.bytes


##########################################################################################




###########################################################################################

if __name__ == "__main__":
    MPC = MultiPartyMediator()
    MPC.start(2)
    MPC.cleanup()