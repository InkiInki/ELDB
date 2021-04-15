import numpy as np
import time as time
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as choose_file
import tkinter.simpledialog as simple_dialog
from Gui.ELDB import ELDB


class Main:

    def __init__(self):
        """
        """
        self.root = tk.Tk()
        self.__initialize()
        self.root.mainloop()

    def __initialize(self):
        """
        The initialize.
        """
        self.root.title("ELDB")
        self.root.geometry("1000x618")
        self.var = tk.StringVar()
        self.start_time = None
        self.text_print_param = None
        self.text_print_middle = None
        self.text_print_final = None
        self.combobox = None
        self.text_choose_file = None
        self.algorithm_type_var = None
        self.classifier_var = None
        self.distance_var = None
        self.performance_var = None
        self.radio_button = None
        self.scale = None
        self.text_learning_rate = None
        self.text_batch_size = None
        self.text_maximum_psi = None
        self.text_maximum_iter = None
        self.file_name = "D:/Data/Benchmark/musk1+.mat"
        self.learning_rate = 0.25
        self.num_batch = 2
        self.maximum_psi = 100
        self.maximum_iter = 1
        self.__choose_file()
        self.__set_parameters()
        self.__set_button()
        self.__set_text()

    def __set_time(self):
        """
        Set time.
        """
        tk.Label(self.root, text="数据集选取", bg="#d3fbfb", fg="red", font=("Times", 32),
                 relief=tk.RAISED, textvariable=self.var). \
            place(relx=0.0, rely=0.9, relheight=0.1, relwidth=0.206)
        self.start_time = time.time()
        self.__get_time()

    def __get_time(self):
        """
        Get time
        """
        self.var.set("%.2f" % (time.time() - self.start_time))
        self.root.after(1000, self.__get_time)

    def __set_button(self):
        """
        Set button.
        """
        tk.Button(self.root, text="Run", fg="red", font=("Times", 20),
                  command=lambda: self.__main__()). \
            place(relx=0.206, rely=0.9, relheight=0.1, relwidth=0.206)
        tk.Button(self.root, text="Break", fg="red", font=("Times", 20),
                  command=lambda: self.root.destroy()). \
            place(relx=0.412, rely=0.9, relheight=0.1, relwidth=0.206)

    def __set_algorithm_type(self):
        """
        Set algorithm type.
        """
        tk.Label(self.root, text="Algorithm type", font=("Times", 14)). \
            place(relx=0.0, rely=0.75, relheight=0.05, relwidth=0.1545)
        self.algorithm_type_var = tk.StringVar()
        ttk.Combobox(self.root, textvariable=self.algorithm_type_var, values=["a", "r"]). \
            place(relx=0.0, rely=0.80, relheight=0.05, relwidth=0.1545)
        self.algorithm_type_var.set("a")

    def __set_classifier(self):
        """
        Set classifiers.
        """
        tk.Label(self.root, text="Classifier", font=("Times", 14)). \
            place(relx=0.1545, rely=0.75, relheight=0.05, relwidth=0.1545)
        self.classifier_var = tk.StringVar()
        ttk.Combobox(self.root, textvariable=self.classifier_var,
                     values=["svm", "knn", "j48"]). \
            place(relx=0.1545, rely=0.8, relheight=0.05, relwidth=0.1545)
        self.classifier_var.set("svm")

    def __set_distance(self):
        """
        Set distance function.
        """
        tk.Label(self.root, text="Distance", font=("Times", 14)). \
            place(relx=0.309, rely=0.75, relheight=0.05, relwidth=0.1545)
        self.distance_var = tk.StringVar()
        ttk.Combobox(self.root, textvariable=self.distance_var,
                     values=["ave_hausdorff", "vir_hausdorff"]). \
            place(relx=0.309, rely=0.8, relheight=0.05, relwidth=0.1545)
        self.distance_var.set("ave_hausdorff")

    def __set_performance_measure(self):
        """
        Set performance measure.
        """
        tk.Label(self.root, text="Performance", font=("Times", 14)). \
            place(relx=0.4635, rely=0.75, relheight=0.05, relwidth=0.1545)
        self.performance_var = tk.StringVar()
        ttk.Combobox(self.root, textvariable=self.performance_var,
                     values=["f1-measure", "accuracy"]). \
            place(relx=0.4635, rely=0.8, relheight=0.05, relwidth=0.1545)
        self.performance_var.set("F1-measure")

    def __set_text(self):
        """
        Set text.
        """
        tk.Label(self.root, text="Parameters display", font=("Times", 14)). \
            place(relx=0.618, rely=0.0, relheight=0.05, relwidth=0.382)
        self.text_print_param = tk.Text(self.root)
        self.text_print_param.place(relx=0.618, rely=0.05, relheight=0.191, relwidth=0.382)
        tk.Label(self.root, text="Final results", font=("Times", 14)). \
            place(relx=0.618, rely=0.191, relheight=0.05, relwidth=0.382)
        self.text_print_final = tk.Text(self.root)
        self.text_print_final.place(relx=0.618, rely=0.241, relheight=0.191, relwidth=0.382)
        tk.Label(self.root, text="Middle results", font=("Times", 14)). \
            place(relx=0.618, rely=0.4, relheight=0.05, relwidth=0.382)
        self.text_print_middle = tk.Text(self.root)
        self.text_print_middle.place(relx=0.618, rely=0.45, relheight=0.618, relwidth=0.382)

    def __choose_file(self):
        """
        Choose file.
        """
        tk.Button(self.root, text="Data selection", font=("Times", 20),
                  command=lambda: self.__get_file()). \
            place(relx=0.0, rely=0.0, relheight=0.1, relwidth=0.618)
        self.text_choose_file = tk.Text(self.root)
        self.text_choose_file.place(relx=0.0, rely=0.1, relheight=0.05, relwidth=0.618)
        self.text_choose_file.insert(0.0, "The default setting: " + self.file_name)

    def __get_file(self):
        """
        Get file and return its name.
        """
        self.file_name = str(choose_file.askopenfilename())
        self.text_choose_file.delete(0.0, tk.END)
        self.text_choose_file.insert(0.0, "The chosen datasets: " + self.file_name)

    def __set_method_type(self):
        """
        Set the method type to "g", "p", "n", "b".
        """
        tk.Label(self.root, text="Algorithm types", font=("Times", 14)). \
            place(relx=0.0, rely=0.20, relheight=0.05, relwidth=0.618)
        self.radio_button = tk.StringVar()
        tk.Radiobutton(self.root, text="g", variable=self.radio_button, value="g"). \
            place(relx=0.0, rely=0.25, relheight=0.05, relwidth=0.1045)
        tk.Radiobutton(self.root, text="p", variable=self.radio_button, value="p"). \
            place(relx=0.17, rely=0.25, relheight=0.05, relwidth=0.1045)
        tk.Radiobutton(self.root, text="n", variable=self.radio_button, value="n"). \
            place(relx=0.34, rely=0.25, relheight=0.05, relwidth=0.1045)
        tk.Radiobutton(self.root, text="b", variable=self.radio_button, value="b"). \
            place(relx=0.51, rely=0.25, relheight=0.05, relwidth=0.1045)
        self.radio_button.set("g")

    def __set_psi_ratio(self):
        """
        Set psi ratio.
        """
        tk.Label(self.root, text="Set the ratio of the number of chosen bags", font=("Times", 14)). \
            place(relx=0.0, rely=0.35, relheight=0.05, relwidth=0.618)
        self.scale = tk.DoubleVar()
        tk.Scale(self.root, orient=tk.HORIZONTAL, from_=0.0, to=1.0, tickinterval=0.1,
                 resolution=0.01, variable=self.scale). \
            place(relx=0.0, rely=0.40, relheight=0.1, relwidth=0.618)
        self.scale.set(0.38)

    def __set_other_param(self):
        """
        Set learning rate, batch size and the maximum psi.
        """
        tk.Label(self.root, text="Other params", font=("Times", 14)). \
            place(relx=0.0, rely=0.50, relheight=0.1, relwidth=0.618)
        tk.Button(self.root, text="Learning rate", command=lambda: self.__ask_learning_rate()). \
            place(relx=0.0, rely=0.60, relheight=0.05, relwidth=0.1545)
        self.text_learning_rate = tk.Text(self.root)
        self.text_learning_rate.place(relx=0.0, rely=0.65, relheight=0.05, relwidth=0.1545)
        self.text_learning_rate.insert(0.0, self.learning_rate)
        tk.Button(self.root, text="Batch size", command=lambda: self.__ask_batch_size()). \
            place(relx=0.1545, rely=0.60, relheight=0.05, relwidth=0.1545)
        self.text_batch_size = tk.Text(self.root)
        self.text_batch_size.place(relx=0.1545, rely=0.65, relheight=0.05, relwidth=0.1545)
        self.text_batch_size.insert(0.0, self.num_batch)
        tk.Button(self.root, text="Maximum psi", command=lambda: self.__ask_maximum_psi()). \
            place(relx=0.309, rely=0.60, relheight=0.05, relwidth=0.1545)
        self.text_maximum_psi = tk.Text(self.root)
        self.text_maximum_psi.place(relx=0.309, rely=0.65, relheight=0.05, relwidth=0.1545)
        self.text_maximum_psi.insert(0.0, self.maximum_psi)
        tk.Button(self.root, text="Maximum iter", command=lambda: self.__ask_maximum_iter()). \
            place(relx=0.4635, rely=0.60, relheight=0.05, relwidth=0.1545)
        self.text_maximum_iter = tk.Text(self.root)
        self.text_maximum_iter.place(relx=0.4635, rely=0.65, relheight=0.05, relwidth=0.1545)
        self.text_maximum_iter.insert(0.0, self.maximum_iter)

    def __ask_learning_rate(self):
        """
        Input the learning rate.
        """
        self.learning_rate = simple_dialog.askfloat("", "Input learning rate:")
        self.text_learning_rate.delete(0.0, tk.END)
        self.text_learning_rate.insert(0.0, self.learning_rate)

    def __ask_batch_size(self):
        """
        Input the batch size.
        """
        self.num_batch = simple_dialog.askinteger("", "Input batch size:")
        self.text_batch_size.delete(0.0, tk.END)
        self.text_batch_size.insert(0.0, self.num_batch)

    def __ask_maximum_psi(self):
        """
        Input the maximum psi.
        """
        self.maximum_psi = simple_dialog.askinteger("", "Input maximum psi:")
        self.text_maximum_psi.delete(0.0, tk.END)
        self.text_maximum_psi.insert(0.0, self.maximum_psi)

    def __ask_maximum_iter(self):
        """
        Input the maximum iter.
        """
        self.maximum_iter = simple_dialog.askinteger("", "Input maximum iter:")
        self.text_maximum_iter.delete(0.0, tk.END)
        self.text_maximum_iter.insert(0.0, self.maximum_iter)

    def __set_parameters(self):
        """
        Set parameters.
        """
        self.__set_method_type()
        self.__set_algorithm_type()
        self.__set_classifier()
        self.__set_distance()
        self.__set_performance_measure()
        self.__set_psi_ratio()
        self.__set_other_param()

    def __text_print_param(self):
        """
        Print params.
        """
        self.text_print_param.delete(0.0, tk.END)
        self.text_print_param.insert(tk.END, "Algorithm type: " + self.algorithm_type_var.get() + "\n")
        self.text_print_param.insert(tk.END, "Bag selection type: " + self.radio_button.get() + "\n")
        self.text_print_param.insert(tk.END, "Ratio for discriminative bags: " + str(self.scale.get()) + "\n")
        self.text_print_param.insert(tk.END,
                                     "Learning rate %.2f, batch size %d\n" % (self.learning_rate, self.num_batch))
        self.text_print_param.insert(tk.END, "Maximum number of discriminative bags: " + str(self.maximum_psi) + "\n")
        self.text_print_param.insert(tk.END, "Maximum iterations: " + str(self.maximum_iter) + "\n")

    def __main__(self):
        """
        The main function.
        """
        self.__set_time()
        self.__text_print_param()
        self.text_print_middle.delete(0.0, tk.END)
        self.text_print_final.delete(0.0, tk.END)
        run = ELDB(self.file_name, m=self.maximum_psi, algorithm_type=[self.radio_button.get()],
                   b2b_type=[self.distance_var.get()], k=10, update_mode=self.algorithm_type_var.get()[0],
                   classifier_type=[self.classifier_var.get()],
                   learning_r=self.learning_rate, num_batch=self.num_batch)
        performance_max = None
        total_time = 0
        for loop in range(self.maximum_iter):
            start_time = time.time()
            performance = np.zeros(10)
            current_time = 0
            self.text_print_middle.insert(tk.END, "The " + str(loop) + "-th loop:")
            for i in range(10):
                performance[i] = run.classify() * 100
                self.text_print_middle.insert(tk.END, "  The %d-th 10CV: %.2f\n" % (i, performance[i]))
            current_time = time.time() - start_time
            total_time += current_time
            self.text_print_middle.insert(tk.END, "The %d-loop time cost: %.2f\n" % (loop, current_time))
            performance_std = np.std(performance, ddof=1)
            performance_ave = (np.sum(performance) / 10)
            self.text_print_middle.insert(tk.END, "Per: %.2f; std: %.2f; time cost %.2f\n" % (
                performance_ave, performance_std, current_time))
            self.text_print_final.insert(tk.END, "The %d -th per: %.2f; std: %.2f; time cost %.2f\n" % (
                loop, performance_ave, performance_std, current_time))
            if performance_max is None or performance_max[0] < performance_ave:
                performance_max = [performance_ave, performance_std]
        self.text_print_final.insert(0.0, "The maximum per: %.2f; std: %.2f; total time: %.2f\n" % (
            performance_max[0], performance_max[1], total_time))


if __name__ == '__main__':
    main = Main()
