import numpy as np
import matplotlib.pyplot as plt



#1/PRINT all characters in “your name” by using for loops.
def print_name_char(name):
    name = name.replace(" ","")

    for char in name:
        print(char)


#2/ Print all odd numbers x such that 1<=x<=10
def print_odd():
    list = [1,2,3,4,5,6,7,8,9,10]

    for num in list:
        if num%2!=0:
            print(num)

#3/ a/Compute the sum of all numbers in 2/
def print_sum():
    total_sum = sum(range(1, 3))
    print(total_sum)
#b/ Compute the sum of all number from 1 to 6
def print_sum1():
    total_sum = sum(range(1, 7))
    print(total_sum)

#4/ Given mydict={“a”: 1,”b”:2,”c”:3,”d”:4}.

# a/ Print all key in mydict
def print_key():
    mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
    print("Tất cả các khóa trong mydict:")
    for key in mydict.keys():
        print(key)

# b/ Print all values in mydict
def print_values():
    mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
    print("\nTất cả các giá trị trong mydict:")
    for value in mydict.values():
        print(value)

# c/ Print all keys and values
def print_key_values():
    mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
    print("\nTất cả các khóa và giá trị trong mydict:")
    for key, value in mydict.items():
        print(f"{key}: {value}")

# 5/ Given courses=[131,141,142,212] and names=[“Maths”,”Physics”,”Chem”, “Bio”].
# Print a sequence of tuples, each of them contains one courses and one names

def print_course_name_tuples():
    # Danh sách các khóa học và tên
    courses = [131, 141, 142, 212]
    names = ["Maths", "Physics", "Chem", "Bio"]
    # Tạo và in ra danh sách các tuple
    course_name_tuples = list(zip(courses, names))
    for course_name in course_name_tuples:
        print(course_name)

#6/ Find the number of consonants in “jabbawocky” by two ways
# 	a/ Directly (i.e without using the command “continue”)
def count_consonants_directly():
    consonants = "bcdfghjklmnpqrstvwxyz"
    word = "jabbawocky"
    count = 0
    for char in word:
        if char in consonants:
            count += 1


    print("Số lượng phụ âm (trực tiếp):", count)
# 	b/ Check whether it’s characters are in vowels set and using the command “continue”
def count_consonants_with_continue():
    vowels = "aeiou"
    word = "jabbawocky"
    count = 0
    for char in word:
        if char in vowels:
            continue  # Bỏ qua nếu là nguyên âm
        count += 1  # Nếu không phải là nguyên âm, đếm là phụ âm


    print("Số lượng phụ âm (sử dụng continue):", count)



#7/ a is a number such that -2<=a<3.
# Print out all the results of 10/a using try…except. When a=0, print out “can’t divided by zero
def calculate_division():
    for a in range(-2, 3):  # A sẽ nhận các giá trị -2, -1, 0, 1, 2
        try:
            result = 10 / a
            print(f"10 / {a} = {result}")
        except ZeroDivisionError:
            print("can't divide by zero")

#8/ Given ages=[23,10,80]
# And names=[Hoa,Lam,Nam]. Using lambda function to sort a list containing tuples (“age”,”name”) by increasing of the ages
def sort_ages_names():
    ages = [23, 10, 80]
    names = ["Hoa", "Lam", "Nam"]

    # Tạo danh sách tuple (tuổi, tên)
    age_name_tuples = list(zip(ages, names))

    # Sắp xếp danh sách tuple theo tuổi
    sorted_tuples = sorted(age_name_tuples, key=lambda x: x[0])

    # In ra kết quả
    print("Danh sách đã sắp xếp theo tuổi:")
    for item in sorted_tuples:
        print(item)
## READ FILES
#9/ Create  a file “firstname.txt”:
# a/ Open this file for reading
# b/Print each line of this file
# c/ Using .read to read the file and Print it
def read_firstname_file():
    try:
        # Mở file để đọc
        with open('firstname.txt', 'r') as file:
            # Đọc dòng đầu tiên
            first_line = file.readline().strip()
            print("Dòng đầu tiên của file:", first_line)

            # Đọc toàn bộ nội dung của file
            content = file.read()
            print("Nội dung của file:")
            print(content)
    except FileNotFoundError:
        print(f"File không tồn tại.")


## DEFINE A FUNCTION
# 1/ Define a function that return the sum of two numbers a and b. Try with a=3, b=4.
def sum_of_two_numbers(a, b):
    return a + b
result = sum_of_two_numbers(3, 4)
print("The sum of 3 and 4 is:", result)

# 2/ Create a 3x3 matrix M=■8(1&2&3@4&5&6@7&8&9) and vector v=■8(1&2&3)
# And check the rank and the shape of this matrix and vector v.
def create_matrix_and_vector():
    # Tạo ma trận 3x3 M
    M = np.array([[4, 5, 6],
                  [7, 8, 9],
                  [1, 2, 3]])

    # Tạo vector v
    v = np.array([1, 2, 3])

    # Kiểm tra hạng (rank) của ma trận M
    rank_M = np.linalg.matrix_rank(M)

    # Hình dạng (shape) của ma trận M và vector v
    shape_M = M.shape
    shape_v = v.shape

    # In ma trận, vector, hạng và hình dạng
    print("Ma trận M:")
    print(M)
    print("\nVector v:")
    print(v)
    print("\nHạng của ma trận M:", rank_M)
    print("Hình dạng của ma trận M:", shape_M)
    print("Hình dạng của vector v:", shape_v)

# 3/ Create a new 3x3 matrix such that its’ elements are the sum of corresponding (position) element of M plus 3.
def create_and_modify_matrix():
    # Tạo ma trận 3x3 M
    M = np.array([[4, 5, 6],
                  [7, 8, 9],
                  [1, 2, 3]])

    # Tạo vector v
    v = np.array([1, 2, 3])

    # Tạo ma trận mới bằng cách cộng 3 vào từng phần tử của M
    new_matrix = M + 3

    # In ma trận, vector và ma trận mới
    print("Ma trận M:")
    print(M)
    print("\nVector v:")
    print(v)
    print("\nMa trận mới (M + 3):")
    print(new_matrix)


# 4/ Create the transpose of M and v
def create_and_modify_matrix():
    # Tạo ma trận 3x3 M
    M = np.array([[4, 5, 6],
                  [7, 8, 9],
                  [1, 2, 3]])

    # Tạo vector v
    v = np.array([1, 2, 3])

    # Tạo ma trận mới bằng cách cộng 3 vào từng phần tử của M
    new_matrix = M + 3

    # Tạo ma trận chuyển vị của M
    transpose_M = M.T

    # Tạo ma trận chuyển vị của vector v (biến thành ma trận cột)
    transpose_v = v.reshape(-1, 1)

    # In ma trận, vector, ma trận mới, và các chuyển vị
    print("Ma trận M:")
    print(M)
    print("\nVector v:")
    print(v)
    print("\nMa trận mới (M + 3):")
    print(new_matrix)
    print("\nChuyển vị của M:")
    print(transpose_M)
    print("\nChuyển vị của v:")
    print(transpose_v)

## MATHS
# 5/ Compute the norm of x=(2,7). Normalization vector x.
def compute_norm_and_normalize():
    # Tạo vector x
    x = np.array([2, 7])

    # Tính chuẩn (norm) của vector x
    norm_x = np.linalg.norm(x)

    # Chuẩn hóa vector x
    normalized_x = x / norm_x

    # In kết quả
    print("Vector x:")
    print(x)
    print("\nChuẩn của vector x:", norm_x)
    print("\nVector x sau khi chuẩn hóa:")
    print(normalized_x)
# 6/ Given a=[10,15], b=[8,2] and c=[1,2,3]. Compute a+b, a-b, a-c. Do all of them work? Why?
def compute_operations():
    # Định nghĩa các vector
    a = np.array([10, 15])
    b = np.array([8, 2])
    c = np.array([1, 2, 3])

    # Tính toán a + b
    sum_ab = a + b

    # Tính toán a - b
    diff_ab = a - b

    # Thử tính toán a - c (sẽ gây lỗi do kích thước khác nhau)
    try:
        diff_ac = a - c
    except ValueError as e:
        diff_ac = str(e)

    # In kết quả
    print("Vector a:", a)
    print("Vector b:", b)
    print("Vector c:", c)
    print("\na + b:", sum_ab)
    print("a - b:", diff_ab)
    print("\na - c:", diff_ac)

# 7/ Compute the dot product of a and b.
def compute_dot_product():
    # Định nghĩa các vector
    a = np.array([10, 15])
    b = np.array([8, 2])

    # Tính tích vô hướng của a và b
    dot_product = np.dot(a, b)

    # In kết quả
    print("Vector a:", a)
    print("Vector b:", b)
    print("\nTích vô hướng của a và b:", dot_product)
# 8/ Given matrix A=[[2,4,9],[3,6,7]].
# 	a/ Check the rank and shape of A
# 	b/ How can get the value 7 in A?
# 	c/ Return the second column of A.
def analyze_matrix():
    # Định nghĩa ma trận A
    A = np.array([[2, 4, 9],
                  [3, 6, 7]])

    # a) Kiểm tra hạng và hình dạng của A
    rank_A = np.linalg.matrix_rank(A)
    shape_A = A.shape

    # b) Lấy giá trị 7 trong A
    value_7 = A[1, 2]  # Giá trị ở hàng 1, cột 2 (0-based index)

    # c) Trả về cột thứ hai của A
    second_column = A[:, 1]  # Tất cả hàng, cột thứ 1

    # In kết quả
    print("Ma trận A:")
    print(A)
    print("\nHạng của ma trận A:", rank_A)
    print("Hình dạng của ma trận A:", shape_A)
    print("\nGiá trị 7 trong A:", value_7)
    print("\nCột thứ hai của A:")
    print(second_column)

#9/ Create a random  3x3 matrix  with the value in range (-10,10).
def create_random_matrix():
    # Tạo ma trận 3x3 với giá trị ngẫu nhiên trong khoảng (-10, 10)
    matrix = np.random.randint(-10, 10, size=(3, 3))
    print("Ma trận ngẫu nhiên 3x3:")
    print(matrix)

#10/ Create an identity (3x3) matrix.
def create_identity_matrix():
    # Tạo ma trận đơn vị 3x3
    identity_matrix = np.eye(3)
    print("Ma trận đơn vị 3x3:")
    print(identity_matrix)

#11/ Create a 3x3 random matrix with the value in range (1,10). Compute the trace of this matrix by 2 ways:
	# a/ By one command
def compute_trace_one_command():
    matrix = np.random.randint(1, 10, size=(3, 3))
    # Tính trace bằng một lệnh
    print("Trace (bằng một lệnh):", np.trace(matrix))

# b/ By using for loops
def compute_trace_with_loops():
    matrix = np.random.randint(1, 10, size=(3, 3))
    # Tính trace bằng vòng lặp
    trace = 0
    for i in range(matrix.shape[0]):  # Duyệt qua các hàng
        trace += matrix[i, i]  # Cộng giá trị đường chéo
    print("Trace (bằng vòng lặp):", trace)

#12/Create a 3x3 diagonal matrix with the value in main diagonal 1,2,3.
def create_diagonal_matrix():
    # Tạo ma trận chéo 3x3 với các giá trị 1, 2, 3 trên đường chéo chính
    diagonal_values = [1, 2, 3]
    #Sử dụng hàm np.diag để tạo ma trận chéo từ danh sách các giá trị [1, 2, 3].
    diagonal_matrix = np.diag(diagonal_values)
    print("Ma trận chéo 3x3:")
    print(diagonal_matrix)

#13/ Given A=[[1,1,2],[2,4,-3],[3,6,-5]]. Compute the determinant of A


def compute_determinant():
    # Tính định thức của ma trận

    # Ma trận A
    A = np.array([[1, 1, 2],
                  [2, 4, -3],
                  [3, 6, -5]])

    # Tính định thức
    print("Định thức của ma trận A:", np.linalg.det(A))

compute_determinant()

#14/ Given a1=[1,-2,-5] and a2=[2,5,6]. Create a matrix M such that the first column is a1 and the second column is a2.
def create_matrix():
    a1 = [1, -2, -5]
    a2 = [2, 5, 6]
    # Tạo ma trận M với a1 là cột đầu tiên và a2 là cột thứ hai
    #Sử dụng hàm np.column_stack để kết hợp hai danh sách thành một ma trận, trong đó mỗi danh sách trở thành một cột.
    M = np.column_stack((a1, a2))
    print("Ma trận M:",M)


#15/ Simply plot the value of the square of y with y in range (-5<=y<6).
# Tạo giá trị y trong khoảng từ -5 đến 5
def plot_square_of_y():
    # Tạo giá trị y trong khoảng từ -5 đến 5
    y = np.arange(-5, 6)

    # Tính bình phương của y
    y_squared = y ** 2
    #Hàm plot_square_of_y sẽ tạo ra các giá trị y, tính bình phương của chúng và vẽ đồ thị.
    # Vẽ đồ thị
    plt.plot(y, y_squared, marker='o')
    plt.title("Đồ thị của y^2")
    plt.xlabel("y")
    plt.ylabel("y^2")
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.xlim(-5, 5)
    plt.ylim(0, 30)
    plt.show()


#16/ Create 4-evenly-spaced values between 0 and 32 (including endpoints)
def create_evenly_spaced_values(  ):
    # Tạo các giá trị cách đều
    start = 0
    end = 32
    num_values = 4
    values = np.linspace(start, end, num_values)
    print("4 giá trị cách đều giữa 0 và 32:", values)

#17/ Get 50 evenly-spaced values from -5 to 5 for x. Calculate y=x**2. Plot (x,y).
def plot_parabola():
    # Tạo 50 giá trị cách đều từ -5 đến 5
    x = np.linspace(-5, 5, 50)

    # Tính y = x^2
    y = x ** 2

    # Vẽ đồ thị
    plt.plot(x, y)
    plt.title("Đồ thị của y = x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.xlim(-5, 5)
    plt.ylim(0, 30)

#18/ Plot y=exp(x) with label and title.
def plot_exponential():
    # Tạo giá trị x từ -2 đến 2
    x = np.linspace(-2, 2, 100)

    # Tính y = exp(x)
    y = np.exp(x)

    # Vẽ đồ thị
    plt.plot(x, y, label='y = exp(x)', color='blue')
    plt.title("Đồ thị của y = e^x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.xlim(-2, 2)
    plt.ylim(0, np.exp(2))
    plt.show()
#19/ Similarly for y=log(x) with x from 0 to 5.
    def plot_logarithm():
        # Tạo giá trị x từ 0.01 đến 5 (bắt đầu từ 0.01 để tránh log(0))
        x = np.linspace(0.01, 5, 100)

        # Tính y = log(x)
        y = np.log(x)

        # Vẽ đồ thị
        plt.plot(x, y, label='y = log(x)', color='green')
        plt.title("Đồ thị của y = log(x)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.axhline(0, color='black', linewidth=0.5, ls='--')
        plt.axvline(0, color='black', linewidth=0.5, ls='--')
        plt.xlim(0, 5)
        plt.ylim(-3, np.log(5))
        plt.show()

#20/ Draw two graphs y=exp(x), y=exp(2*x) in the same graph and y=log(x) and y=log(2*x) in the same graph using subplot.
def plot_exponential_and_logarithm():
    # Tạo giá trị x cho đồ thị hàm số mũ
    x_exp = np.linspace(-2, 2, 100)
    y_exp1 = np.exp(x_exp)
    y_exp2 = np.exp(2 * x_exp)

    # Tạo giá trị x cho đồ thị hàm số log
    x_log = np.linspace(0.01, 5, 100)
    y_log1 = np.log(x_log)
    y_log2 = np.log(2 * x_log)

    # Tạo subplot cho hàm số mũ
    plt.subplot(1, 2, 1)
    plt.plot(x_exp, y_exp1, label='y = exp(x)', color='blue')
    plt.plot(x_exp, y_exp2, label='y = exp(2x)', color='orange')
    plt.title("Đồ thị của y = exp(x) và y = exp(2x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')

    # Tạo subplot cho hàm số log
    plt.subplot(1, 2, 2)
    plt.plot(x_log, y_log1, label='y = log(x)', color='green')
    plt.plot(x_log, y_log2, label='y = log(2x)', color='red')
    plt.title("Đồ thị của y = log(x) và y = log(2x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')

    # Hiển thị đồ thị
    plt.tight_layout()
    plt.show()