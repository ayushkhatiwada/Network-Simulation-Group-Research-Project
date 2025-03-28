import time
from passive_monitoring.passive_monitoring_interface.general_sketch import Sketch

class WaveSketch(Sketch):
    """
    A basic implementation of WaveSketch.
    
    For each packet, this sketch determines the current time window
    (using the configured bin size) and accumulates a count.
    When a new window begins, the current count is appended to a signal array.
    
    A full WaveSketch would perform online wavelet transform and compression,
    but here we simply record raw counts.
    """
    def __init__(self, start_time, bin_size):
        self.start_time = start_time
        self.bin_size = bin_size
        self.w0 = None  # initial window index
        self.current_window = None
        self.current_count = 0
        self.signal = []  # list of counts for finished windows

    def process_packet(self, packet, switch_id):
        now = time.time()
        window_id = int((now - self.start_time) // self.bin_size)
        if self.w0 is None:
            self.w0 = window_id
            self.current_window = window_id
        if window_id == self.current_window:
            self.current_count += 1
        else:
            self.signal.append(self.current_count)
            self.current_window = window_id
            self.current_count = 1

    def record_event(self):
        self.process_packet(None, None)

    def get_signal(self):
        # Return the complete signal, including the current window count.
        return self.signal + [self.current_count]
