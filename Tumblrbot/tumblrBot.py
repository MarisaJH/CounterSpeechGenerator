import pytumblr
from keyFunctions import createKeys
def gainAcess():
    # get Access
    keys = createKeys()
    client = pytumblr.TumblrRestClient(
    keys[0], keys[1], keys[2], keys[3]
    )
    return client
def retreiveName(client):
# get the name of account
    info = client.info()
    name = info['user']['name']
    return name 
def postOnAccount(client, title, post):
    name = retreiveName(client)
    client.create_text(name, state = 'published', title = title, body = post)
    print('Done.')

def getPosts(client, blogName):
    posts = client.posts(blogName, 'text')
    posts = posts['posts']
    postname = posts[0]['body']
    postid = posts[0]['id']
    print(postname)
    print(postid)

def justMessingAround():
    client = gainAcess()
    name = retreiveName(client)
    followers = client.followers(name)
    follower = followers['users'][0]['name']
    info = client.blog_info(follower)
    #name = info['user']['name']
    getPosts(client, name)
    #print(info)
    
def guide():
    client = gainAcess()
    finshed = False
    while finshed != True:
        task = input('What would you like to do? Get Name? Or Post?: ')
        if task == 'Get Name' or task == 'get name':
            print (retreiveName(client))
            
        elif task == 'Post' or task == 'post':
            post = input('What would you like to post?: ')
            title = input('What is the title?: ')
            postOnAccount(client, title, post)
        else:
            print('I could not understand your request')
        done = input('Is that all today?: ')
        if done == 'Yes' or done == 'yes':
            finshed = True
#guide()
justMessingAround()
        
    
    