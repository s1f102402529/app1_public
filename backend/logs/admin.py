from django.contrib import admin
from .models import BabyLog

@admin.register(BabyLog)
class BabyLogAdmin(admin.ModelAdmin):
    # 管理画面の一覧に表示する列
    list_display = ('log_type', 'timestamp', 'memo', 'created_at')
    # 右側に絞り込みフィルターを表示
    list_filter = ('log_type', 'timestamp')
    # メモの内容で検索できるようにする
    search_fields = ('memo',)
    exclude = ('user',)

    def get_queryset(self, request):
        return super().get_queryset(request).filter(user=request.user)

    def save_model(self, request, obj, form, change):
        if not change:
            obj.user = request.user
        super().save_model(request, obj, form, change)