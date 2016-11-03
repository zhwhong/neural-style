import Image
import os

tests_dir = "./output/"
def main():

        filetest = os.listdir(tests_dir)
        num_tests = len(filetest)
        for j in range(num_tests):
                filename = os.path.join(tests_dir,filetest[j])
                fname,ftyle = os.path.splitext(os.path.basename(filename))
                im = Image.open(filename)
                r,g,b = im.split()
                r.save("./tunnel/"+fname+"_r.jpg")
                g.save("./tunnel/"+fname+"_g.jpg")
                b.save("./tunnel/"+fname+"_b.jpg")

if __name__ == '__main__':
    main()