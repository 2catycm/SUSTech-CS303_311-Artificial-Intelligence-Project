#include <cstdint>
#include <vector>
#include <array>
enum ChessColor { 
    None = 0,
    White = 1, 
    Black = -1, 
};
template<size_t N=8>
struct Point{
    size_t x, y;
    static constexpr std::array<Point, 8> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    bool isValid(){
        return x<N && y<N;
    }
    Point operator+(Point other){
        return Point(x+other.x, y+other.y);
    }
    Point operator-(Point other){
        return Point(x-other.x, y-other.y);
    }
};
/**
 * @brief 
 * 
 * @tparam n chessboard size
 */
template<size_t N = 8>
struct ChessBoard{
    static constexpr size_t max_piece_cnt = N*N;
    ChessColor board[N][N]{};
    auto actions(ChessColor color)-> std::vector<Point<N>> const{
        auto result = std::vector<Point<N>>();
        for (size_t i = 0; i < N; i++){
            for (size_t j = 0; j < N; j++){
                auto point = Point<N>{i, j};
                if (this->isValidMove(color, point)){
                    result.push_back(point);
                }
            }  
        }
        return result;
    }
    bool isValidMove(ChessColor color, Point<N> point) const{
        if (board[point.x][point.y]==ChessColor::None)
            return false;
        for(const auto& direction:Point::directions){
            auto neighbor = direction+point;
            if (!neighbor.isValid() || int(board[point.x][point.y])!=-int(color)){
                
            }
        }
        return true;
    }
    ChessColor operator [](Point<N> point) const{
        return this->board[point.x][point.y];
    }
    ChessColor& operator [](Point<N> point){
        return this->board[point.x][point.y];
    }
    ChessBoard(){
        auto color = ChessColor::Black;
        int locations[2] = {N/2-1, N/2};
        for (int i = 0; i <2; i++){
            for (int j = 0; j < 2; j++){
                j^=i;
                board[locations[i]][locations[j]] = color;
                color*=-1;
            }
        }
    }
};