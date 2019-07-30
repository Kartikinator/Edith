from pytube import YouTube
from urllib.request import Request, urlopen

req = Request('https://youtube.com', headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()


yt = YouTube(str(input("Enter the video link: ")))
videos = yt.streams.all()
print(str(yt))

s = 1
for v in videos:
    print(str(s)+". "+str(v))
    s += 1

n = int(input("Enter the number of the video: "))
vid = videos[n-1]

destination = str(input("Enter the destination: "))
vid.download(destination)

print(yt.filename+"\nHas been successfully downloaded")
