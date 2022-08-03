#include "../rule.hpp"
#include <cstdint>
#include <iostream>
int main(int argc, char const *argv[]){
    auto chessboard = ChessBoard<8>();
    std::cout << chessboard.max_piece_cnt<<std::endl;
    std::cout << int(chessboard[{0, 0}])<<std::endl;
    chessboard[{0, 0}] = ChessColor::Black;
    std::cout << int(chessboard[{0, 0}])<<std::endl;
    return 0;
}