#include "../rule.hpp"
#include <cstdint>
#include <iostream>
int main(int argc, char const *argv[]){
    auto a = ChessColor::Black;
    std::cout << int(a)<<std::endl;
    std::cout << ChessBoard<8>::max_piece_cnt<<std::endl;
    return 0;
}
