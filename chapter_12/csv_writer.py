"""A simple class to write statistics to csv file."""
import os
import csv


class CsvWriter:
    """A logging object writing to a CSV file.
    """

    def __init__(self, fname):
        """Initializes a `CsvWriter`.
        Args:
          fname: File name (path) for file to be written to.
        """
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fname = fname
        self._header_written = False
        self._fieldnames = None

    def write(self, values):
        """Appends given values as new row to CSV file."""

        if self._fieldnames is None:
            self._fieldnames = values.keys()
        # Open a file in 'append' mode, so we can continue logging safely to the
        # same file after e.g. restarting from a checkpoint.
        with open(self._fname, 'a') as file:
            # Always use same fieldnames to create writer, this way a consistency
            # check is performed automatically on each write.
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self):
        """Closes the `CsvWriter`."""
        pass

    def get_state(self):
        """Retrieves `CsvWriter` state as a `dict` (e.g. for serialization)."""
        return {
            'header_written': self._header_written,
            'fieldnames': self._fieldnames,
        }

    def set_state(self, state):
        """Sets `CsvWriter` state from a (potentially de-serialized) dictionary."""
        self._header_written = state['header_written']
        self._fieldnames = state['fieldnames']
