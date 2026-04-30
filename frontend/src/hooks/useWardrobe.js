import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import { toast } from 'sonner'

export function useWardrobe() {
  const queryClient = useQueryClient()

  const { data: items = [], isLoading, error } = useQuery({
    queryKey: ['wardrobe'],
    queryFn: api.getWardrobe,
  })

  const addItem = useMutation({
    mutationFn: api.addToWardrobe,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe'] })
      toast.success('Item added to wardrobe')
    },
  })

  // We have the API call for delete, but maybe locally we also want an option 
  // Let's implement full remove if API supports it, or just local.
  const removeItem = useMutation({
    mutationFn: api.removeFromWardrobe,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['wardrobe'] })
      toast.success('Item removed')
    },
  })

  return {
    items,
    isLoading,
    error,
    addItem: addItem.mutate,
    isAdding: addItem.isPending,
    removeItem: removeItem.mutate,
    isRemoving: removeItem.isPending
  }
}
