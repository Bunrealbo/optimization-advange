import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        Initializes a DataLoader object with the given data.

        Args:
            X (ndarray): The input data.
            y (ndarray): The target data.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        if shuffle:
            indices = np.random.permutation(len(y))
            self.X, self.y = X[indices], y[indices]
        else:
            self.X, self.y = X, y

        self.batch_size = batch_size
        self.dim = X.shape[1]

        # Calculate the number of batches
        self.n_batches = len(y) // batch_size
        if len(y) % batch_size != 0:
            self.n_batches += 1
        
    def __iter__(self):
        """
        Initializes an iterator for the DataLoader object.

        Returns:
            self: The DataLoader object itself, allowing it to be used in a for loop.
        """
        self.current_batch = 0
        return self
    
    def __len__(self):
        """
        Returns the number of batches in the DataLoader.

        Returns:
            int: The number of batches in the DataLoader.
        """
        return self.n_batches

    def __next__(self):
        """
        Returns the next batch of data from the DataLoader.

        This method is used to iterate over the DataLoader object and retrieve
        batches of data for training. It advances the internal counter to the
        next batch and returns the corresponding input and target data.

        Returns:
            tuple: A tuple containing the input data (X) and target data (y) for
                   the current batch.

        Raises:
            StopIteration: If the current batch index is greater than or equal to
                            the total number of batches.

        """
        if self.current_batch >= self.n_batches:
            raise StopIteration()
        
        start = self.current_batch * self.batch_size
        end = (self.current_batch + 1) * self.batch_size
        self.current_batch += 1

        return self.X[start:end], self.y[start:end]


