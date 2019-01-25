#define PORT 1153
#define PORT_S 1994
#define BUFSIZE 2048
#define _BSD_SOURCE

#include <netinet/in.h>	/* needed for sockaddr_in */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <cstring>	/* for memcpy */
#include <string>
#include <iostream>
using namespace std;

int do_search(const char* my_message);
int check_first(string target);
void show_interface();

int main(){
	
	struct sockaddr_in myaddr;      /* our address */
    struct sockaddr_in remaddr;     /* remote address */
    socklen_t addrlen = sizeof(remaddr);            /* length of addresses */
    int recvlen;                    /* # bytes received */
    int socket_v;                         /* our socket */
    unsigned char buf[BUFSIZE];     /* receive buffer */
    

    if ((socket_v = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            cerr << "cannot create socket" << endl;
            return 0;
    }
    memset((char *)&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET;
    myaddr.sin_port = htons(PORT);
    inet_aton("172.16.199.204", &myaddr.sin_addr);

    if (bind(socket_v, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
            cerr << "bind failed" << endl;
            return 0;
    }
    while(1){
    	show_interface();
		recvlen = recvfrom(socket_v, buf, BUFSIZE, 0, (struct sockaddr *)&remaddr, &addrlen);
		cout << "received" <<  recvlen << "bytes" << endl;
		string show_info(buf, buf + recvlen);
		if (recvlen > 0) {
		        cout << "received message: \n" << show_info << endl;
		}
    }
    
	
	return 0;
}





int do_search(const char* my_message){
	struct sockaddr_in servaddr;    /* server address */
	int fd;
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT_S);
	inet_aton("172.16.199.204", &servaddr.sin_addr);
	if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
                cerr << "cannot create socket" << endl;
                return 0;
        }
	if (sendto(fd, my_message, strlen(my_message), 0, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
		cerr << "sendto failed" << endl;
		return 0;
	}
	return 0;
}





int check_first(string target){
	for(int i = 0; i < target.length(); i++){
		if(target[i] < 'a' || target[i] > 'z'){
			return 0;
		}
	}
	return 1;
}






void show_interface(){
	string target;
	cout << "Please input the word which you wanna know:"<<endl;
	while(true){
		cin >> target;
		if(target.length() == 1){
			if(target[0]>='a'&&target[0]<='z'){
				cout << "It is a letters" << endl;
			}
			else{
				cout << "Fail to search that word!" << endl;
				cout << "Please input the word again!" << endl;
			}
		}
		else{
			if(check_first(target)){
				const char* p =  target.data();
				do_search(p);
				break;
			}
		}
	}
}