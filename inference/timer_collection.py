import time

class Timer:
    def __init__(self, purpose: str):
        self.purpose: str = purpose
        self.times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args): # *args is necessary to just catch other arguments passed by the system and that we do not need
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.times.append(self.elapsed_time)

    def get_average_time(self):
        return sum(self.times) / len(self.times) if self.times else -1


class TimerCollection:
    def __init__(self):
        self.timers = {}
        self.purposes = []

    def add_timer(self, purpose: str):
        self.purposes.append(purpose)
        self.timers[purpose] = Timer(purpose)
        return self.timers[purpose]

    def get_timer(self, purpose: str):
        if purpose in self.purposes:
            return self.timers[purpose]
        else:
            return self.add_timer(purpose)
    
    def save_timer_results_as_csv(self, output_filename):
        with open(output_filename, 'w') as f:
            f.write("Purpose,Average Time\n")
            for purpose in self.purposes:
                timer = self.get_timer(purpose)
                average_time = timer.get_average_time()
                f.write(f"{purpose},{average_time}\n")

