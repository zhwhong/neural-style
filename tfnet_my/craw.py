import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import requests
import urllib
import re
import random
from time import sleep

outfile = './data/train/people/'  #save image

def main():
	i=1
	for x in xrange(0,40,20): #(,60--control pages,)20=1pages 40=2pages 60=3pages....
		#use baidutupian (oldversion!...go top-right to change)
		url='http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%E8%B5%B5%E5%98%89%E6%95%8F&pn='+str(x)
		headers={}
		content = requests.get(url).text
		imgs=re.findall('"objURL":"(.*?)"',content)  
		for img in imgs:
			print "iteration : ",x, "    image : ",i
			if img.split('.')[-1] != 'jpg':
				continue   	#only accept jpg
			try:
				path=outfile+str(i)+".jpg"	#save image
				urllib.urlretrieve(img,path)
				#print u'NO. '+str(i)+u' images : ',img
				i+=1
			except:
				pass
	


if __name__=='__main__':
	main() 
