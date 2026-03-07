```mermaid
graph TD
    %% 定义流程图方向：TD (Top to Down) 从上到下

    Start((开始爬塔)) --> Battle[战斗楼层]
    Start((开始爬塔)) --> Shop[商店楼层]
    Start((开始爬塔)) --> Select[选项楼层]

    %% 战斗楼层
    subgraph BattleFloor[战斗楼层]
        Battle --> SelectCard[选择buff]
        SelectCard --> SelectCard
    end
    SelectCard --> Start

    %% 选择楼层
    subgraph SelectFloor[选择楼层]
        Select --> Dialog[对话]
        Select --> Selection[选项]
        Dialog --> Selection
        Selection --> NoteGet[音符获得界面]
        Selection --> SelectCard
    end
    NoteGet --> Start
    SelectCard --> Start
    Selection --> Start
    
    %% 商店楼层
    subgraph ShopFloor[商店楼层]
        Shop --> Purchase[购买]
        Shop --> Enhance[强化]
        Shop --> LastShop{是否最后一层？}
        
        Purchase --> PurchaseForLoop[8个商品购买循环]
        PurchaseForLoop --> ShopRefresh{刷新次数&金币?}
        ShopRefresh -- 是 --> PurchaseForLoop
        ShopRefresh -- 否 --> Shop
        
        Enhance --> EnhanceMoney{强化所需金币?}
        EnhanceMoney -- 是 --> SelectCard
        EnhanceMoney -- 否 --> Shop
        SelectCard --> Enhance
    end
    LastShop -- 否 --> Start
    LastShop -- 是 --> End((结束爬塔))
