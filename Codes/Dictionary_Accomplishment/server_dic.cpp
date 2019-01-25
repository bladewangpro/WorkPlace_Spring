#define PORT 1995
// #define PORT_S 1153
#define PORT_S 1994
#define BUFSIZE 2048
#define _BSD_SOURCE

#include <cstring>
#include <netinet/in.h>	
#include <sys/socket.h>
#include <arpa/inet.h>



#include <iostream>
#include <string>
#include <fstream>
using namespace std;

int find_target(ifstream & stream, string target);
string show_difinition(ifstream & stream, string target);
// int send_message(const char* message);
int send_message(string message);

int do_server();

int main(){
	do_server();
	return 0;
}

int do_server(){
	struct sockaddr_in myaddr;      /* our address */
    struct sockaddr_in remaddr;     /* remote address */
    socklen_t addrlen = sizeof(remaddr);            /* length of addresses */
    int recvlen;                    /* # bytes received */
    int fd;                         /* our socket */
    unsigned char buf[BUFSIZE];     /* receive buffer */

    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            cerr << "cannot create socket" << endl;
            return 0;
    }
    memset((char *)&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET;
    myaddr.sin_port = htons(PORT);
    inet_aton("192.168.1.11", &myaddr.sin_addr);

    if (bind(fd, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
            cerr << "bind failed" << endl;
            return 0;
    }
    while(true){
            cout << "waiting on port " <<  PORT << endl;
            recvlen = recvfrom(fd, buf, BUFSIZE, 0, (struct sockaddr *)&remaddr, &addrlen);
            cout << "received" <<  recvlen << "bytes" << endl;
            if (recvlen > 0) {
                    cout << "received message: " << buf << endl;

                    // string target = *buf;
                    string target(buf, buf + recvlen);
                    string mess;
                    const char* p;
                    cout << "_____________+++++++++++"<<target << "----------" << target.length()<<endl;

                    ifstream word_handle("words.txt");
					if(word_handle.is_open()){
						cout << "open the file" << endl;
						if(find_target(word_handle, target)){
							cout << "Bingo! find it!" << endl;
							word_handle.close();
							ifstream dic_handle("Oxford_English_Dictionary.txt");
							if(dic_handle.is_open()){
								target[0] += 'A'-'a';
								target = target + " ";
								mess = show_difinition(dic_handle, target);
								cout <<  mess << endl;
								// p = mess.data();
								// cout<<"++++++++++++++" << p << endl;
								// send_message(p);
								send_message(mess);
							}
							dic_handle.close();
						}
						else{
							word_handle.close();

							mess = "Fail to search that word! \nPlease input the word again!";
							// p = mess.data();
							// send_message(p);
							send_message(mess);
							// cout << "Fail to search that word!" << endl;
							// cout << "Please input the word again!" << endl;
						}
					}



            }
    }
}
int send_message(string message){
	struct sockaddr_in servaddr;    /* server address */
	int fd;
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT_S);
	inet_aton("192.168.1.13", &servaddr.sin_addr);
	// inet_aton("172.16.199.204", &servaddr.sin_addr);
	if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
                cerr << "cannot create socket" << endl;
                return 0;
        }
       cout << "Message data" << message.data() << "message length" << message.length() << endl;
	// if (sendto(fd, message, strlen(message), 0, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
	if (sendto(fd, message.data(), message.length(), 0, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {	
		cerr << "sendto failed" << endl;
		return 0;
	}
	return 0;
}

/*************************************************
Function: int find_target(ifstream & stream, string target)
Description: to check whether the specific word is in the file or not
Input: stream --- the file handle of the opened file, target -- the target word,,, the word obtained from user's input.
Return: 0 --- didn't find approporiate word, 1 --- find the location of the right word.
*************************************************/
int find_target(ifstream & stream, string target){
	string line;
	while(getline(stream, line)){
		if(line.find(target) != string::npos){
			return 1;
		}
	}
	return 0;
}
/*************************************************
Function: int show_difinition(ifstream & stream, string target)
Description: to show the definition of the specific word via dictionary.txt
Input: stream --- the file handle of the opened file, target -- the target word,,, the word obtained from user's input and the first 
character has been changed into capital format to fit the format of dictionary.
Return: return function is designed to terminate the loop
*************************************************/
string show_difinition(ifstream & stream, string target){
	string line, lines;
	int found  = 0;
	// cout << "The definition has been shown as followed:"<<endl;
	while(!stream.eof()){
		getline(stream, line);
		if(target.substr(0, target.size()).compare(line.substr(0, target.size()))==0){
			// cout << line << endl;
			lines += line;
			found = 1;
			getline(stream, line);
		}
		else if(found == 1){
			return lines;
		}
		else{
			continue;
		}
	}
}