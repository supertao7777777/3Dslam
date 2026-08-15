

#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Quaternion.h>
#include <std_msgs/Float64MultiArray.h>
int PORT;
int BUFFER_SIZE;
std::string SERVER_IP;
ros::Publisher qrPosePub;

void handleReceivedData(const std::vector<uint8_t> &receivedData)
{
    if (receivedData.size() < 21)
    {
        ROS_ERROR("Received data length is too short for parsing.");
        return;
    }

    uint8_t result = receivedData[0];
    double x_offset, y_offset, yaw;
    int tag_number;
    for (size_t i = 1; i <= 19; i++)
    {
        result ^= receivedData[i];
    }

    bool on_qrcode = (receivedData[1] & 0x40) == 0x40;
    unsigned int x_position = (receivedData[2] * 0x80 * 0x4000) + (receivedData[3] * 0x4000) + (receivedData[4] * 0x80) + receivedData[5];
    if (x_position > 0x800000)
    {
        x_position = (0x1000000 - x_position);
        x_offset = -static_cast<double>(x_position) / 10000;
    }
    else
    {
        x_offset = static_cast<double>(x_position) / 10000;
    }
    unsigned int y_position = (receivedData[6] * 0x80) + receivedData[7];
    if (y_position > 0x2000)
    {
        y_position = (0x4000 - y_position);
        y_offset = -static_cast<double>(y_position) / 10000;
    }
    else
    {
        y_offset = static_cast<double>(y_position) / 10000;
    }

    unsigned int angle = (receivedData[10] * 0x80) + receivedData[11];
    yaw = static_cast<double>(angle) / 10;
    // 传感器的输出yaw是以Z朝向地面的
    // 输入landmark的四元数中，Z轴是朝向下的，刚好传感器本身Z轴也是朝下的
    // yaw = 360 - yaw;  // 通过360 - yaw 可以等价将角度的方向转换成Z朝上
    if (yaw > 180.0)
    {
        yaw = yaw - 360;
    }
    yaw = yaw * (M_PI / 180.0);

    int raw_tag_number = (receivedData[14] * 0x80 * 0x4000) + (receivedData[15] * 0x4000) + (receivedData[16] * 0x80) + receivedData[17];
    if (raw_tag_number < 0x3FFF)
    {
        raw_tag_number = (receivedData[16] * 0x80) + receivedData[17];
    }
    tag_number = static_cast<int>(raw_tag_number);

    if (result == receivedData[20])
    {
        if (on_qrcode)
        {

            std::cout << "on_qrcode: " << on_qrcode << std::endl;
            std::cout << "tag_number: " << tag_number << std::endl;
            std::cout << "qr_x_offset: " << x_offset << std::endl;
            std::cout << "qr_y_offset: " << y_offset << std::endl;
            std::cout << "qr_theta (yaw): " << yaw << std::endl;
            // ======  发布 topic ======
            std_msgs::Float64MultiArray msg;
            msg.data.resize(4);
            msg.data[0] = static_cast<id_t>(tag_number); // tag_number
            msg.data[1] = x_offset;                        // x
            msg.data[2] = y_offset;                        // y
            msg.data[3] = yaw;                             // yaw (rad)

            qrPosePub.publish(msg);
            // ============================
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "DM_Reader");
    ros::NodeHandle nh;
    //
    nh.param<int>("/DM_Reader/PORT", PORT, 5004);
    nh.param<int>("/DM_Reader/BUFFER_SIZE", BUFFER_SIZE, 1024);
    nh.param<std::string>("/DM_Reader/SERVER_IP", SERVER_IP, "192.168.16.6");
    qrPosePub = nh.advertise<std_msgs::Float64MultiArray>("qr_pose", 10);
    int serverSocket, newSocket;
    struct sockaddr_in serverAddr, clientAddr;
    socklen_t addrLen = sizeof(clientAddr);
    if ((serverSocket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        ROS_ERROR("socket failed");
        return -1;
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(SERVER_IP.c_str());
    serverAddr.sin_port = htons(PORT);

    int opt = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
    {
        ROS_ERROR("bind failed");
        close(serverSocket);
        return -1;
    }

    if (listen(serverSocket, 3) < 0)
    {
        ROS_ERROR("listen failed");
        close(serverSocket);
        return -1;
    }

    volatile bool first_flag = true;

    while (ros::ok())
    {
        if ((newSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &addrLen)) < 0)
        {
            ROS_ERROR("accept failed");
            continue;
        }

        if (first_flag)
        {
            ROS_INFO("DM Reader Connected!");
            first_flag = false;
        }

        uint8_t buffer[21];
        ssize_t bytesRead = recv(newSocket, reinterpret_cast<char *>(buffer), sizeof(buffer), 0);
        if (bytesRead <= 0)
        {
            close(newSocket);
            continue;
        }

        std::vector<uint8_t> receivedData(buffer, buffer + bytesRead);
        handleReceivedData(receivedData);

        close(newSocket);
    }
    ros::spin();
    close(serverSocket);
    return 0;
}
